/**
 * The metadata an ONNX file carries about itself, read from its bytes.
 *
 * The export writes three entries into each model's `metadata_props`: `run`
 * (the training run the weights come from), `model` (the network) and
 * `frame_lags` (the history frames it stacks, as JSON). That is how the page
 * can tell which model a file holds. onnxruntime-web reads a model's inputs
 * and outputs, but not this metadata, so it is read here from the file.
 *
 * An ONNX file is one protobuf `ModelProto` message. Its field 14 is a
 * repeated `StringStringEntryProto`, each a `key` (field 1) and a `value`
 * (field 2). Protobuf frames every field with its number and its wire type, so
 * the reader needs no schema: it opens field 14 and steps over everything else,
 * the graph and its weights included. A file cut off inside a field, or framed
 * in a way no ONNX file is, fails with an error rather than a partial answer.
 */

/** ModelProto.metadata_props, and StringStringEntryProto's key and value. */
const METADATA_PROPS = 14;
const KEY = 1;
const VALUE = 2;

/** Protobuf's wire types: how a field's payload is framed. */
const VARINT = 0;
const FIXED64 = 1;
const DELIMITED = 2;
const FIXED32 = 5;

const UTF8 = new TextDecoder("utf-8", { fatal: true });

class Reader {
  private at: number;

  constructor(
    private readonly bytes: Uint8Array,
    start: number,
    private readonly end: number,
  ) {
    this.at = start;
  }

  done(): boolean {
    return this.at >= this.end;
  }

  /** An unsigned varint. Values past 2^53 lose precision, which only matters
   * for numbers the reader steps over. */
  varint(): number {
    let value = 0;
    for (let i = 0; i < 10; i += 1) {
      if (this.at >= this.end) throw new Error("the file ends inside a number");
      const byte = this.bytes[this.at] as number;
      this.at += 1;
      value += (byte & 0x7f) * 2 ** (7 * i);
      if (byte < 0x80) return value;
    }
    throw new Error("a number runs past ten bytes");
  }

  /** The next field's number and wire type. */
  tag(): { field: number; wire: number } {
    const tag = this.varint();
    const field = Math.floor(tag / 8);
    if (field === 0) throw new Error("field 0 is not a field");
    return { field, wire: tag % 8 };
  }

  /** The next `length` bytes, as a reader of their own. */
  take(length: number): Reader {
    if (!Number.isSafeInteger(length) || length > this.end - this.at) {
      throw new Error("the file ends inside a field");
    }
    const part = new Reader(this.bytes, this.at, this.at + length);
    this.at += length;
    return part;
  }

  /** A length-delimited payload, as a reader. */
  delimited(): Reader {
    return this.take(this.varint());
  }

  text(): string {
    return UTF8.decode(this.bytes.subarray(this.at, this.end));
  }

  skip(wire: number): void {
    if (wire === VARINT) this.varint();
    else if (wire === FIXED64) this.take(8);
    else if (wire === DELIMITED) this.delimited();
    else if (wire === FIXED32) this.take(4);
    else throw new Error(`wire type ${wire} is not one an ONNX file uses`);
  }
}

/** One StringStringEntryProto: its key and its value (empty when absent, as
 * protobuf has it). */
function readEntry(reader: Reader): [string, string] {
  let key = "";
  let value = "";
  while (!reader.done()) {
    const { field, wire } = reader.tag();
    if (field !== KEY && field !== VALUE) {
      reader.skip(wire);
      continue;
    }
    if (wire !== DELIMITED) throw new Error(`a metadata ${field === KEY ? "key" : "value"} must be text`);
    const text = reader.delimited().text();
    if (field === KEY) key = text;
    else value = text;
  }
  return [key, value];
}

/** The `metadata_props` of the ONNX model in `bytes`, by key. Throws if the
 * file is cut off or malformed, or names a key twice. */
export function modelMetadata(bytes: Uint8Array): ReadonlyMap<string, string> {
  const reader = new Reader(bytes, 0, bytes.length);
  const entries = new Map<string, string>();
  while (!reader.done()) {
    const { field, wire } = reader.tag();
    if (field !== METADATA_PROPS) {
      reader.skip(wire);
      continue;
    }
    if (wire !== DELIMITED) throw new Error("the model's metadata must be a list of entries");
    const [key, value] = readEntry(reader.delimited());
    if (entries.has(key)) throw new Error(`the model names "${key}" twice in its metadata`);
    entries.set(key, value);
  }
  return entries;
}
