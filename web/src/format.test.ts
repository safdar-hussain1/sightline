import { describe, expect, it } from "vitest";
import { NO_VALUE, fmtDeg, fmtPct, fmtRatio } from "./format.ts";

describe("fmtDeg", () => {
  it("keeps the sign with a true minus", () => expect(fmtDeg(-7.04)).toBe("−7.0°"));
  it("never prints a measured non-zero as zero", () => expect(fmtDeg(0.04)).not.toMatch(/^0\.0°$/));
  it("prints angles beyond half a turn", () => expect(fmtDeg(501.78)).toBe("501.8°"));

  it("prints a true zero as zero, never as a negative zero", () => {
    expect(fmtDeg(0)).toBe("0.0°");
    expect(fmtDeg(-0)).toBe("0.0°");
  });

  it("widens a small angle rather than rounding it away, keeping its sign", () => {
    expect(fmtDeg(0.04)).toBe("0.04°");
    expect(fmtDeg(-0.04)).toBe("−0.04°");
    expect(fmtDeg(0.004)).toBe("0.004°");
  });

  it("states a bound below a thousandth of a degree instead of printing noise", () => {
    expect(fmtDeg(0.0002)).toBe("<0.001°");
    expect(fmtDeg(-0.0002)).toBe("−<0.001°");
  });

  it("honours the requested number of decimals", () => {
    expect(fmtDeg(-338.84, 0)).toBe("−339°");
    expect(fmtDeg(2.5, 2)).toBe("2.50°");
    expect(fmtDeg(0.3, 0)).toBe("0.3°");
  });

  it("uses U+2212, never the hyphen-minus", () => {
    expect(fmtDeg(-12.3)).not.toContain("-");
    expect(fmtDeg(-12.3).codePointAt(0)).toBe(0x2212);
  });

  it("prints an em dash where there is no measurement", () => {
    expect(fmtDeg(Number.NaN)).toBe(NO_VALUE);
    expect(fmtDeg(Number.POSITIVE_INFINITY)).toBe(NO_VALUE);
  });
});

describe("fmtRatio", () => {
  it("uses a multiplication sign", () => expect(fmtRatio(4.26)).toBe("4.3×"));

  it("never calls a ratio that differs from one an even 1.0×", () => {
    expect(fmtRatio(0.96)).toBe("0.96×");
    expect(fmtRatio(1.04)).toBe("1.04×");
    expect(fmtRatio(1)).toBe("1.0×");
  });

  it("never prints a measured non-zero ratio as zero", () => {
    expect(fmtRatio(0.03)).toBe("0.03×");
  });

  it("prints an em dash where there is no ratio", () => {
    expect(fmtRatio(Number.NaN)).toBe(NO_VALUE);
  });
});

describe("fmtPct", () => {
  it("reads a fraction as a percentage", () => {
    expect(fmtPct(0.6431)).toBe("64%");
    expect(fmtPct(0.6431, 1)).toBe("64.3%");
  });

  it("keeps the sign of a negative fraction with a true minus", () => {
    expect(fmtPct(-0.12)).toBe("−12%");
  });

  it("never prints a measured non-zero as 0%", () => {
    expect(fmtPct(0.004)).toBe("0.4%");
    expect(fmtPct(0.0004)).toBe("0.04%");
    expect(fmtPct(0.00004)).toBe("<0.01%");
    expect(fmtPct(0)).toBe("0%");
  });

  it("never rounds a share short of all of it up to 100%", () => {
    expect(fmtPct(0.9996)).toBe("99.96%");
    expect(fmtPct(1)).toBe("100%");
  });

  it("prints an em dash where there is no share", () => {
    expect(fmtPct(Number.NaN)).toBe(NO_VALUE);
  });
});

describe("every formatter, across many magnitudes", () => {
  // A sweep rather than a list of cases: the defect this guards against is a
  // rounding rule that erases a measurement somewhere nobody thought to look.
  const values: number[] = [];
  for (let exponent = -7; exponent <= 3; exponent += 1) {
    for (const mantissa of [1, 1.5, 2.5, 4.9999, 5, 7.25, 9.95]) {
      values.push(mantissa * 10 ** exponent, -mantissa * 10 ** exponent);
    }
  }

  it("prints no measured non-zero as a zero", () => {
    const zero = /^−?0(\.0+)?(°|×|%)$/;
    for (const value of values) {
      for (const text of [fmtDeg(value), fmtDeg(value, 0), fmtDeg(value, 2), fmtRatio(value), fmtPct(value)]) {
        expect(text, `${value} printed as ${text}`).not.toMatch(zero);
      }
    }
  });

  it("keeps the sign of every negative value", () => {
    for (const value of values.filter((v) => v < 0)) {
      for (const text of [fmtDeg(value), fmtRatio(value), fmtPct(value)]) {
        expect(text.startsWith("−"), `${value} printed as ${text}`).toBe(true);
      }
    }
  });
});
