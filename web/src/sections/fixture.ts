/**
 * A results.json for tests: the keys the page reads, in the shape the
 * site-data builder writes them, with the values of the reports this site
 * ships. Each call returns a fresh copy, so a test may break one key without
 * touching another. Nothing in the page imports this file.
 */

const RUNS = (model: string, protocol: string) => [0, 1, 2].map((seed) => `drive2018_${model}_${protocol}_s${seed}`);

export function resultsFixture(): Record<string, unknown> {
  return {
    assets: {
      models: { sightline: "models/sightline.onnx", shuffled: "models/sightline-shuffled.onnx" },
      clips: { c2017: "data/clips/c2017/clip.json", c2018: "data/clips/c2018/clip.json" }
    },
    shipped: { run: "drive2018_sightlinenet_blocked_s2", model: "sightlinenet", protocol: "blocked", seed: 2 },
    twin: { run: "drive2018_sightlinenet_leaky_s2" },
    baselines: {
      run: "drive2018_sightlinenet_blocked_s2",
      test: {
        n: 6000,
        model: {
          mae_deg: 10.12272448789344,
          n: 6000
        },
        baselines: {
          extrap: {
            mae_deg: 0.6901967614466945,
            n: 6000
          },
          mean: {
            mae_deg: 17.84318643687274,
            n: 6000
          },
          zero: {
            mae_deg: 17.479270016382138,
            n: 6000
          },
          zoh: {
            mae_deg: 0.5927266640439629,
            n: 6000
          }
        }
      },
      cross_drive: {
        drive: "drive2017",
        n: 45404,
        model: {
          mae_deg: 9.771067024628094,
          n: 45404
        },
        baselines: {
          extrap: {
            mae_deg: 0.48579622978518006,
            n: 45404
          },
          mean: {
            mae_deg: 15.473928367779282,
            n: 45404
          },
          zero: {
            mae_deg: 15.52269723393061,
            n: 45404
          },
          zoh: {
            mae_deg: 0.5068826531183521,
            n: 45404
          }
        }
      }
    },
    timing: {
      drive2017: null,
      drive2018: {
        closest_eval_s: {
          blocked: {
            test: {
              eval_frame: 61249,
              gap_s: 24.613,
              train_frame: 61750
            },
            val: {
              eval_frame: 35749,
              gap_s: 24.286,
              train_frame: 36250
            }
          },
          forward: {
            test: {
              eval_frame: 54501,
              gap_s: 535.169,
              train_frame: 44426
            },
            val: {
              eval_frame: 44927,
              gap_s: 26.441,
              train_frame: 44426
            }
          }
        },
        frame_interval_s: {
          max: 0.338,
          mean: 0.053,
          median: 0.057,
          min: 0.025,
          p05: 0.029,
          p95: 0.091
        },
        lag_gap_s: [
          {
            lag: 2,
            max: 0.557,
            mean: 0.105,
            median: 0.1,
            min: 0.053,
            p05: 0.066,
            p95: 0.157
          }
        ]
      }
    },
    cutoffs: {
      regime_bins_deg: [5.0, 30.0],
      sign_min_abs_deg: 5.0,
      tolerances_deg: [1.5, 5.0, 15.0]
    },
    drives: {
      drive2017: { frames: 45406, median_dt_s: null, repaired_labels: 268 },
      drive2018: { frames: 63825, median_dt_s: 0.056999921798706055, repaired_labels: 368 }
    },
    paired_leakage: {
      pilotnet: {
        ratio: {
          mean: 3.166602797825647,
          sd: 0.13786572388950946,
          values: [3.3195783025911982, 3.051959717769372, 3.1282703731163717]
        },
        shuffled_mae_deg: {
          mean: 4.130170764717941,
          sd: 0.28707933891936716,
          values: [4.032252277606631, 3.9048609355625907, 4.453399080984602]
        },
        blocked_mae_deg: {
          mean: 13.07809728495099,
          sd: 1.0415481408090872,
          values: [13.385377171316913, 11.91747827882825, 13.931436404707808]
        },
        difference_deg: {
          mean: 8.947926520233048,
          sd: 0.8124058229619574,
          values: [9.353124893710282, 8.01261734326566, 9.478037323723207]
        },
        per_seed: [
          {
            blocked_mae_deg: 13.385377171316913,
            difference_deg: 9.353124893710282,
            n: 893,
            ratio: 3.3195783025911982,
            seed: 0,
            shuffled_mae_deg: 4.032252277606631,
            zero_mae_deg: 17.308191551773707
          },
          {
            blocked_mae_deg: 11.91747827882825,
            difference_deg: 8.01261734326566,
            n: 889,
            ratio: 3.051959717769372,
            seed: 1,
            shuffled_mae_deg: 3.9048609355625907,
            zero_mae_deg: 16.160995532040925
          },
          {
            blocked_mae_deg: 13.931436404707808,
            difference_deg: 9.478037323723207,
            n: 943,
            ratio: 3.1282703731163717,
            seed: 2,
            shuffled_mae_deg: 4.453399080984602,
            zero_mae_deg: 18.825535512678442
          }
        ]
      },
      sightlinenet: {
        ratio: {
          mean: 3.0465626341557503,
          sd: 0.6142642724803958,
          values: [3.747038174282365, 2.5997789697772737, 2.792870758407613]
        },
        shuffled_mae_deg: {
          mean: 3.3481055871804877,
          sd: 0.40005183380705683,
          values: [2.9803185991667553, 3.28994326465276, 3.774054897721948]
        },
        blocked_mae_deg: {
          mean: 10.086980146092957,
          sd: 1.3648407093489716,
          values: [11.167367562601575, 8.553125311204633, 10.540447564472663]
        },
        difference_deg: {
          mean: 6.738874558912468,
          sd: 1.4621276864914683,
          values: [8.187048963434819, 5.263182046551873, 6.766392666750715]
        },
        per_seed: [
          {
            blocked_mae_deg: 11.167367562601575,
            difference_deg: 8.187048963434819,
            n: 893,
            ratio: 3.747038174282365,
            seed: 0,
            shuffled_mae_deg: 2.9803185991667553,
            zero_mae_deg: 17.308191551773707
          },
          {
            blocked_mae_deg: 8.553125311204633,
            difference_deg: 5.263182046551873,
            n: 889,
            ratio: 2.5997789697772737,
            seed: 1,
            shuffled_mae_deg: 3.28994326465276,
            zero_mae_deg: 16.160995532040925
          },
          {
            blocked_mae_deg: 10.540447564472663,
            difference_deg: 6.766392666750715,
            n: 943,
            ratio: 2.792870758407613,
            seed: 2,
            shuffled_mae_deg: 3.774054897721948,
            zero_mae_deg: 18.825535512678442
          }
        ]
      }
    },
    groups: {
      "pilotnet/blocked": {
        runs: RUNS("pilotnet", "blocked"),
        cross_mae_deg: { mean: 19.283728156069824, sd: 1.2222151247645154 },
        cross_r: { mean: 0.3697796405644785, sd: 0.022896068977211026 },
        cross_skill: { mean: -0.24229235843871808, sd: 0.07873729071342774 },
        cross_zero_mae_deg: 15.52269723393061,
        test_mae_deg: { mean: 12.937989008557295, sd: 0.15543423650517776 },
        test_r: { mean: 0.571845406585894, sd: 0.00800882975760467 },
        test_skill: { mean: 0.2598095345840305, sd: 0.0088924901531643 },
        test_zero_mae_deg: 17.479270016382138,
        val_mae_deg: { mean: 11.529227264213944, sd: 0.546047727284138 }
      },
      "pilotnet/forward": {
        runs: RUNS("pilotnet", "forward"),
        cross_mae_deg: { mean: 18.230527678843572, sd: 2.6421114927392297 },
        cross_r: { mean: 0.33153453822883755, sd: 0.08169097544090065 },
        cross_skill: { mean: -0.1744432944935621, sd: 0.17020956170966955 },
        cross_zero_mae_deg: 15.52269723393061,
        test_mae_deg: { mean: 13.625845819999489, sd: 4.143022383114607 },
        test_r: { mean: 0.08725015995222235, sd: 0.1490698567806062 },
        test_skill: { mean: -0.4485684827442407, sd: 0.440446173159765 },
        test_zero_mae_deg: 9.406421568820829,
        val_mae_deg: { mean: 7.649069464043262, sd: 0.6961133761713462 }
      },
      "pilotnet/leaky": {
        runs: RUNS("pilotnet", "leaky"),
        cross_mae_deg: { mean: 21.74953238502313, sd: 2.239228637475938 },
        cross_r: { mean: 0.36162398413269736, sd: 0.036307717827238586 },
        cross_skill: { mean: -0.4011438899601458, sd: 0.1442551254933501 },
        cross_zero_mae_deg: 15.52269723393061,
        test_mae_deg: { mean: 4.005480440160848, sd: 0.21386362833981673 },
        test_r: { mean: 0.9246003166837632, sd: 0.009454788472508609 },
        test_skill: { mean: 0.6951777264965134, sd: 0.009588771929433299 },
        test_zero_mae_deg: 13.13605197527287,
        val_mae_deg: { mean: 3.934114750744758, sd: 0.19253158242847027 }
      },
      "sightlinenet-1f/blocked": {
        runs: RUNS("sightlinenet-1f", "blocked"),
        cross_mae_deg: { mean: 15.229990120248688, sd: 0.6559645157668917 },
        cross_r: { mean: 0.4758464267713509, sd: 0.027175466209321767 },
        cross_skill: { mean: 0.01885671731341254, sd: 0.04225841075692945 },
        cross_zero_mae_deg: 15.52269723393061,
        test_mae_deg: { mean: 11.417595442926304, sd: 0.5376786988146927 },
        test_r: { mean: 0.6521484377445405, sd: 0.02997046077885373 },
        test_skill: { mean: 0.3467922040093572, sd: 0.030760935571723703 },
        test_zero_mae_deg: 17.479270016382138,
        val_mae_deg: { mean: 10.280497225492468, sd: 0.09754151420085072 }
      },
      "sightlinenet/blocked": {
        runs: RUNS("sightlinenet", "blocked"),
        cross_mae_deg: { mean: 10.640418845640982, sd: 1.304190160711699 },
        cross_r: { mean: 0.6028904074029199, sd: 0.07679068339833359 },
        cross_skill: { mean: 0.3145251314711981, sd: 0.08401826957372512 },
        cross_zero_mae_deg: 15.52269723393061,
        test_mae_deg: { mean: 10.238193261658825, sd: 0.6843360253560705 },
        test_r: { mean: 0.6661287035386055, sd: 0.009055063224882684 },
        test_skill: { mean: 0.41426654247784617, sd: 0.03915129320130011 },
        test_zero_mae_deg: 17.479270016382138,
        val_mae_deg: { mean: 7.69308831234504, sd: 0.658414139044221 }
      },
      "sightlinenet/forward": {
        runs: RUNS("sightlinenet", "forward"),
        cross_mae_deg: { mean: 8.76265019157143, sd: 0.3848312635355173 },
        cross_r: { mean: 0.8205465606948695, sd: 0.008059720285638278 },
        cross_skill: { mean: 0.4354943564564664, sd: 0.024791520296764267 },
        cross_zero_mae_deg: 15.52269723393061,
        test_mae_deg: { mean: 6.180404257216079, sd: 0.24567487148382644 },
        test_r: { mean: 0.7853493612262751, sd: 0.017868471363496213 },
        test_skill: { mean: 0.34295904005598987, sd: 0.026117782377323755 },
        test_zero_mae_deg: 9.406421568820829,
        val_mae_deg: { mean: 6.606190071346602, sd: 0.3494818269927485 }
      },
      "sightlinenet/leaky": {
        runs: RUNS("sightlinenet", "leaky"),
        cross_mae_deg: { mean: 7.958739434335296, sd: 0.1316228160313178 },
        cross_r: { mean: 0.837166411331428, sd: 0.0035586066767067694 },
        cross_skill: { mean: 0.48728372947077, sd: 0.008479377910148722 },
        cross_zero_mae_deg: 15.52269723393061,
        test_mae_deg: { mean: 2.9048510941506365, sd: 0.0874730019907449 },
        test_r: { mean: 0.9844214860210413, sd: 0.0012395459187010205 },
        test_skill: { mean: 0.7788633029556049, sd: 0.0004335637885169679 },
        test_zero_mae_deg: 13.13605197527287,
        val_mae_deg: { mean: 2.9113827617443064, sd: 0.08407327486981431 }
      }
    },
    ladder: {
      pilotnet: [
        {
          drive: "drive2018",
          mae_deg: {
            mean: 4.005480440160848,
            sd: 0.21386362833981673,
            values: [3.7665348755813555, 4.070949385526614, 4.178957059374575]
          },
          protocol: "leaky",
          rung: "shuffled",
          seeds: [0, 1, 2],
          split: "test",
          zero_mae_deg: 13.13605197527287
        },
        {
          drive: "drive2018",
          mae_deg: {
            mean: 12.937989008557295,
            sd: 0.15543423650517776,
            values: [13.01658509740179, 13.038429265941348, 12.758952662328747]
          },
          protocol: "blocked",
          rung: "blocked",
          seeds: [0, 1, 2],
          split: "test",
          zero_mae_deg: 17.479270016382138
        },
        {
          drive: "drive2018",
          mae_deg: {
            mean: 13.625845819999489,
            sd: 4.143022383114607,
            values: [15.065162773726238, 8.955122993239405, 16.857251693032822]
          },
          protocol: "forward",
          rung: "later in the drive",
          seeds: [0, 1, 2],
          split: "test",
          zero_mae_deg: 9.406421568820829
        },
        {
          drive: "drive2017",
          mae_deg: {
            mean: 19.283728156069824,
            sd: 1.2222151247645154,
            values: [20.51000439511419, 19.275565036336513, 18.06561503675877]
          },
          protocol: "blocked",
          rung: "another day",
          seeds: [0, 1, 2],
          split: "cross_drive",
          zero_mae_deg: 15.52269723393061
        }
      ],
      sightlinenet: [
        {
          drive: "drive2018",
          mae_deg: {
            mean: 2.9048510941506365,
            sd: 0.0874730019907449,
            values: [2.828082914139468, 2.886388803615307, 3.0000815646971355]
          },
          protocol: "leaky",
          rung: "shuffled",
          seeds: [0, 1, 2],
          split: "test",
          zero_mae_deg: 13.13605197527287
        },
        {
          drive: "drive2018",
          mae_deg: {
            mean: 10.238193261658825,
            sd: 0.6843360253560705,
            values: [10.972918060079216, 9.618937237003818, 10.12272448789344]
          },
          protocol: "blocked",
          rung: "blocked",
          seeds: [0, 1, 2],
          split: "test",
          zero_mae_deg: 17.479270016382138
        },
        {
          drive: "drive2018",
          mae_deg: {
            mean: 6.180404257216079,
            sd: 0.24567487148382644,
            values: [6.066055253625127, 6.012746885132623, 6.462410632890487]
          },
          protocol: "forward",
          rung: "later in the drive",
          seeds: [0, 1, 2],
          split: "test",
          zero_mae_deg: 9.406421568820829
        },
        {
          drive: "drive2017",
          mae_deg: {
            mean: 10.640418845640982,
            sd: 1.304190160711699,
            values: [12.140027868494577, 10.010161643800277, 9.771067024628094]
          },
          protocol: "blocked",
          rung: "another day",
          seeds: [0, 1, 2],
          split: "cross_drive",
          zero_mae_deg: 15.52269723393061
        }
      ],
      "sightlinenet-1f": [
        {
          drive: "drive2018",
          mae_deg: {
            mean: 11.417595442926304,
            sd: 0.5376786988146927,
            values: [11.18797344127049, 11.032852957703174, 12.031959929805248]
          },
          protocol: "blocked",
          rung: "blocked",
          seeds: [0, 1, 2],
          split: "test",
          zero_mae_deg: 17.479270016382138
        },
        {
          drive: "drive2017",
          mae_deg: {
            mean: 15.229990120248688,
            sd: 0.6559645157668917,
            values: [14.474027342018376, 15.566987922490293, 15.6489550962374]
          },
          protocol: "blocked",
          rung: "another day",
          seeds: [0, 1, 2],
          split: "cross_drive",
          zero_mae_deg: 15.52269723393061
        }
      ]
    },
    regimes: {
      models: {
        pilotnet: {
          regimes: {
            balanced_mae_deg: 20.93981525965682,
            gentle: { mae_deg: 8.873407215126047, n: 2416 },
            sharp: { mae_deg: 48.207778250242576, n: 853 },
            straight: { mae_deg: 5.738260313601846, n: 2731 }
          },
          run: "drive2018_pilotnet_blocked_s1",
          seed: 1
        },
        sightlinenet: {
          regimes: {
            balanced_mae_deg: 15.888266038214596,
            gentle: { mae_deg: 5.523734191249877, n: 2416 },
            sharp: { mae_deg: 36.046986396958374, n: 853 },
            straight: { mae_deg: 6.094077526435531, n: 2731 }
          },
          run: "drive2018_sightlinenet_blocked_s2",
          seed: 2
        },
        "sightlinenet-1f": {
          regimes: {
            balanced_mae_deg: 18.364941241574588,
            gentle: { mae_deg: 8.550253160893053, n: 2416 },
            sharp: { mae_deg: 40.244341995584556, n: 853 },
            straight: { mae_deg: 6.3002285682461565, n: 2731 }
          },
          run: "drive2018_sightlinenet-1f_blocked_s2",
          seed: 2
        }
      },
      protocol: "blocked",
      selected_by: "val_mae_deg",
      split: "test",
      zero: {
        balanced_mae_deg: 31.258043210756572,
        gentle: { mae_deg: 12.601645282562206, n: 2416 },
        sharp: { mae_deg: 78.40895083249663, n: 853 },
        straight: { mae_deg: 2.763533517210879, n: 2731 }
      }
    },
    legacy: {
      recipe: "PilotNet in TensorFlow 1: shuffled 80/20 frame split, MSE in radians plus L2, atan output head limited to ±180°",
      trained_on: "drive2017",
      splits: { block: 1500, firewall: 250 },
      seen_first_frame: 2,
      atan_cap: {
        labels_beyond_180: {
          drive2017: { frac: 0.004911245209884156, n: 223 },
          drive2018: { frac: 0.003681942812377595, n: 235 }
        },
        max_abs_pred_deg: 169.84153747558594
      },
      eval: {
        drive2017_all_seen: {
          model: { mae_deg: 4.480583195894129, n: 45404 },
          zero_mae_deg: 15.52269723393061
        },
        drive2018_blocked_test: {
          model: { mae_deg: 21.97078830103064, n: 6000 },
          zero_mae_deg: 17.479270016382138
        },
        drive2018_forward_test: {
          model: { mae_deg: 19.394295458684024, n: 9324 },
          zero_mae_deg: 9.406421568820829
        }
      }
    },
    parity: {
      shuffled: {
        angle_tolerance_deg: 0.0005,
        max_abs_angle_deg: 0.00012969970703125,
        max_abs_saliency: 8.165836334228516e-06,
        onnxruntime: "1.28.0",
        run: "drive2018_sightlinenet_leaky_s2",
        saliency_tolerance: 0.0001,
        stacks: 848
      },
      sightline: {
        angle_tolerance_deg: 0.0005,
        max_abs_angle_deg: 0.000179290771484375,
        max_abs_saliency: 7.450580596923828e-06,
        onnxruntime: "1.28.0",
        run: "drive2018_sightlinenet_blocked_s2",
        saliency_tolerance: 0.0001,
        stacks: 848
      }
    },
    protocol_names: { blocked: "blocked", forward: "later in the drive", leaky: "shuffled" },
    split_rules: {
      blocked: { block: 1500, firewall: 250, label: "blocked", pattern: "TTVTTET" },
      forward: { firewall: 250, label: "later in the drive", test_frac: 0.15, train_frac: 0.7, val_frac: 0.15 },
      leaky: { label: "shuffled", test_frac: 0.15, train_frac: 0.7, val_frac: 0.15 }
    },
    split_sizes: {
      blocked: {
        drive: "drive2018",
        dropped: 12000,
        frames: 63825,
        label: "blocked",
        runs: 9,
        seeds: [0, 1, 2],
        test: 6000,
        train: 39825,
        val: 6000
      },
      forward: {
        drive: "drive2018",
        dropped: 1000,
        frames: 63825,
        label: "later in the drive",
        runs: 6,
        seeds: [0, 1, 2],
        test: 9324,
        train: 44427,
        val: 9074
      },
      leaky: {
        drive: "drive2018",
        dropped: 0,
        frames: 63825,
        label: "shuffled",
        runs: 6,
        seeds: [0, 1, 2],
        test: 9574,
        train: 44677,
        val: 9574
      }
    },
    leakage: {
      gaps: {
        blocked: {
          test: { frac_within_1: 0.0, median_frames: 750.5, min_frames: 501, p95_frames: 975.0500000000002 }
        },
        forward: {
          test: { frac_within_1: 0.0, median_frames: 14736.5, min_frames: 10075, p95_frames: 18931.85 }
        },
        leaky: {
          test: { frac_within_1: 0.9093377898475037, median_frames: 1.0, min_frames: 1, p95_frames: 2.0 }
        }
      },
      memoriser: {
        leaky: { mae_deg: 0.6379842974604311, n: 893 },
        blocked: { mae_deg: 21.867592464485867, n: 893 }
      }
    },
    generated_from: ["reports/summary.json", "reports/runs/drive2018_sightlinenet_blocked_s2.json", "reports/runs/drive2018_sightlinenet_leaky_s2.json", "reports/runs/drive2018_pilotnet_blocked_s0.json", "reports/runs/drive2018_pilotnet_blocked_s1.json", "reports/runs/drive2018_pilotnet_blocked_s2.json", "reports/runs/drive2018_pilotnet_forward_s0.json", "reports/runs/drive2018_pilotnet_forward_s1.json", "reports/runs/drive2018_pilotnet_forward_s2.json", "reports/runs/drive2018_pilotnet_leaky_s0.json", "reports/runs/drive2018_pilotnet_leaky_s1.json", "reports/runs/drive2018_pilotnet_leaky_s2.json", "reports/runs/drive2018_sightlinenet-1f_blocked_s0.json", "reports/runs/drive2018_sightlinenet-1f_blocked_s1.json", "reports/runs/drive2018_sightlinenet-1f_blocked_s2.json", "reports/runs/drive2018_sightlinenet_blocked_s0.json", "reports/runs/drive2018_sightlinenet_blocked_s1.json", "reports/runs/drive2018_sightlinenet_forward_s0.json", "reports/runs/drive2018_sightlinenet_forward_s1.json", "reports/runs/drive2018_sightlinenet_forward_s2.json", "reports/runs/drive2018_sightlinenet_leaky_s0.json", "reports/runs/drive2018_sightlinenet_leaky_s1.json", "reports/leakage.json", "reports/legacy.json"]
  };
}

/** Set the value at a dotted path of `json` (group keys hold a slash, never a
 * dot); `undefined` removes the key. Returns `json`. */
export function patch(json: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const keys = path.split(".");
  let at: Record<string, unknown> = json;
  for (const key of keys.slice(0, -1)) at = at[key] as Record<string, unknown>;
  const last = keys[keys.length - 1] as string;
  if (value === undefined) delete at[last];
  else at[last] = value;
  return json;
}

/** Give `model` the skill `skill` on the rung made of `protocol`'s `split`
 * frames, consistently: each training's error is scaled so their mean is
 * steering straight's times (1 - skill), and the group's error and skill, and
 * the ladder's, are written from those same errors. Returns `json`. */
export function setSkill(json: Record<string, unknown>, model: string, protocol: string, split: "test" | "cross_drive", skill: number): Record<string, unknown> {
  const group = (json["groups"] as Record<string, Record<string, unknown>>)[`${model}/${protocol}`] as Record<string, unknown>;
  const prefix = split === "test" ? "test" : "cross";
  const zero = group[`${prefix}_zero_mae_deg`] as number;
  const rung = ((json["ladder"] as Record<string, Array<Record<string, unknown>>>)[model] ?? []).find(
    (entry) => entry["protocol"] === protocol && entry["split"] === split,
  ) as Record<string, unknown>;
  const error = rung["mae_deg"] as { mean: number; sd: number; values: number[] };
  const factor = (zero * (1 - skill)) / error.mean;
  const values = error.values.map((v) => v * factor);
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const sd = Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (values.length - 1));
  rung["mae_deg"] = { mean, sd, values };
  group[`${prefix}_mae_deg`] = { mean, sd };
  group[`${prefix}_skill`] = { mean: 1 - mean / zero, sd: sd / zero };
  return json;
}
