import { describe, expect, it } from "vitest";
import { NO_VALUE, fmtCount, fmtDeg, fmtMs, fmtPct, fmtRatio, fmtSpread, steerParts } from "./format.ts";

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
    // −0.0002 lies above −0.001: the comparison turns with the sign.
    expect(fmtDeg(-0.0002)).toBe(">−0.001°");
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

describe("steerParts", () => {
  it("says which way the wheel is turned: clockwise is right, as the labels have it", () => {
    expect(steerParts(12.64)).toEqual({ size: "12.6°", side: "right" });
    expect(steerParts(-3)).toEqual({ size: "3.0°", side: "left" });
    expect(steerParts(87.93)).toEqual({ size: "87.9°", side: "right" });
  });

  it("gives a wheel that is not turned no side, and a missing angle no number", () => {
    expect(steerParts(0)).toEqual({ size: "0.0°", side: "" });
    expect(steerParts(Number.NaN)).toEqual({ size: NO_VALUE, side: "" });
  });

  it("keeps the side of an angle too small to print", () => {
    expect(steerParts(-0.0002)).toEqual({ size: "<0.001°", side: "left" });
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

  it("states a bound for a ratio too close to one for three decimals", () => {
    expect(fmtRatio(0.9999)).toBe("<1.000×");
    expect(fmtRatio(1.0001)).toBe(">1.000×");
    expect(fmtRatio(0.996)).toBe("0.996×");
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

  it("states a bound for a share too close to all of it for two decimals", () => {
    expect(fmtPct(0.99999)).toBe(">99.99%");
    expect(fmtPct(1.00001)).toBe(">100.00%");
  });

  it("prints an em dash where there is no share", () => {
    expect(fmtPct(Number.NaN)).toBe(NO_VALUE);
  });
});

describe("fmtCount", () => {
  it("groups the thousands of a count with commas", () => {
    expect(fmtCount(6000)).toBe("6,000");
    expect(fmtCount(45404)).toBe("45,404");
    expect(fmtCount(1234567)).toBe("1,234,567");
    expect(fmtCount(893)).toBe("893");
    expect(fmtCount(0)).toBe("0");
  });

  it("prints only whole counts, with a true minus, and an em dash for anything else", () => {
    expect(fmtCount(-1500)).toBe("−1,500");
    expect(fmtCount(2.5)).toBe(NO_VALUE);
    expect(fmtCount(Number.NaN)).toBe(NO_VALUE);
  });
});

describe("fmtSpread", () => {
  it("prints a standard deviation as a plus-or-minus with the figure's decimals", () => {
    expect(fmtSpread(0.6142)).toBe("±0.6");
    expect(fmtSpread(0.13787, 2)).toBe("±0.14");
    expect(fmtSpread(1.3648, 0)).toBe("±1");
  });

  it("never prints a spread that was measured as none", () => {
    expect(fmtSpread(0.04)).toBe("±0.04");
    expect(fmtSpread(0.0002)).toBe("±<0.001");
    expect(fmtSpread(0)).toBe("±0.0");
  });

  it("refuses a negative or missing spread", () => {
    expect(fmtSpread(-0.5)).toBe(NO_VALUE);
    expect(fmtSpread(Number.NaN)).toBe(NO_VALUE);
  });
});

describe("fmtMs", () => {
  it("prints a time in milliseconds", () => {
    expect(fmtMs(2.53)).toBe("2.5 ms");
    expect(fmtMs(114.28, 0)).toBe("114 ms");
  });

  it("never prints a time that was measured as no time at all", () => {
    expect(fmtMs(0.03)).toBe("0.03 ms");
    expect(fmtMs(0.00002)).toBe("<0.001 ms");
  });

  it("prints an em dash where there is no time", () => {
    expect(fmtMs(Number.NaN)).toBe(NO_VALUE);
    expect(fmtMs(-1)).toBe(NO_VALUE);
  });
});

describe("every formatter, across many magnitudes", () => {
  // A sweep rather than a list of cases: the defect this guards against is a
  // rounding rule that misstates a measurement somewhere nobody thought to look.
  const values: number[] = [];
  for (let exponent = -7; exponent <= 3; exponent += 1) {
    for (const mantissa of [1, 1.5, 2.5, 4.9999, 5, 7.25, 9.95]) {
      values.push(mantissa * 10 ** exponent, -mantissa * 10 ** exponent);
    }
  }
  // ...and the neighbourhood of the round figures a ratio and a share must not
  // be printed as: one, and all of it.
  for (let exponent = -1; exponent >= -8; exponent -= 1) {
    for (const near of [1 - 10 ** exponent, 1 + 10 ** exponent]) {
      values.push(near, -near);
    }
  }

  /** Every formatter's text for `value`, with the number it should state. */
  const printed = (value: number) => [
    { text: fmtDeg(value), stated: value },
    { text: fmtDeg(value, 0), stated: value },
    { text: fmtDeg(value, 2), stated: value },
    { text: fmtRatio(value), stated: value },
    { text: fmtPct(value), stated: value * 100 },
    { text: fmtPct(value, 1), stated: value * 100 },
  ];

  /** The comparison and the number a text states, with U+2212 read as a minus. */
  const parse = (text: string) => {
    const match = /^([<>]?)(−?)(\d+(?:\.\d+)?)(°|×|%)$/.exec(text);
    expect(match, `unparseable: ${text}`).not.toBeNull();
    const [, operator = "", minus = "", digits = ""] = match ?? [];
    const decimals = digits.includes(".") ? digits.length - digits.indexOf(".") - 1 : 0;
    return { operator, number: (minus ? -1 : 1) * Number(digits), decimals };
  };

  it("prints no measured non-zero as a zero", () => {
    const zero = /^−?0(\.0+)?(°|×|%)$/;
    for (const value of values) {
      for (const { text } of printed(value)) {
        expect(text, `${value} printed as ${text}`).not.toMatch(zero);
      }
    }
  });

  it("prints no ratio other than one as an unqualified one", () => {
    for (const value of values.filter((v) => Math.abs(v) !== 1)) {
      expect(fmtRatio(value), `${value}`).not.toMatch(/^−?1(\.0+)?×$/);
    }
  });

  it("prints no share other than all of it as an unqualified 100%", () => {
    for (const value of values.filter((v) => Math.abs(v) !== 1)) {
      for (const text of [fmtPct(value), fmtPct(value, 1)]) {
        expect(text, `${value}`).not.toMatch(/^−?100(\.0+)?%$/);
      }
    }
  });

  it("never shows 100 at all for a share short of all of it", () => {
    for (const value of values.filter((v) => Math.abs(v) < 1)) {
      for (const text of [fmtPct(value), fmtPct(value, 1)]) {
        expect(text, `${value}`).not.toMatch(/^[<>]?−?100\b/);
      }
    }
  });

  it("states only true things: a bound holds, and a figure is the value rounded", () => {
    for (const value of values) {
      for (const { text, stated } of printed(value)) {
        const { operator, number, decimals } = parse(text);
        if (operator === "<") {
          expect(stated, text).toBeLessThan(number);
        } else if (operator === ">") {
          expect(stated, text).toBeGreaterThan(number);
        } else {
          expect(Math.abs(stated - number), `${value} printed as ${text}`).toBeLessThanOrEqual(
            0.5 * 10 ** -decimals + 1e-12,
          );
        }
      }
    }
  });

  it("keeps the sign of every negative value", () => {
    for (const value of values.filter((v) => v < 0)) {
      for (const { text } of printed(value)) {
        expect(text.replace(/^[<>]/, "").startsWith("−"), `${value} printed as ${text}`).toBe(true);
      }
    }
  });

  it("prints no measured spread or time as none, and states only true bounds", () => {
    for (const value of values.filter((v) => v > 0)) {
      for (const [text, unit] of [
        [fmtSpread(value).replace(/^±/, ""), ""],
        [fmtSpread(value, 2).replace(/^±/, ""), ""],
        [fmtMs(value), " ms"],
      ] as const) {
        const match = /^([<>]?)(\d+(?:\.\d+)?)(.*)$/.exec(text);
        expect(match, `unparseable: ${text}`).not.toBeNull();
        const [, operator = "", digits = "", rest = ""] = match ?? [];
        expect(rest).toBe(unit);
        expect(Number(digits) === 0 && operator !== "<", `${value} printed as ${text}`).toBe(false);
        if (operator === "<") expect(value).toBeLessThan(Number(digits));
      }
    }
  });

  it("never uses a space the font cannot draw", () => {
    for (const value of [...values, 0, Number.NaN]) {
      for (const { text } of printed(value)) {
        expect(text).not.toMatch(/[\u2009\u202f]/);
      }
    }
  });
});
