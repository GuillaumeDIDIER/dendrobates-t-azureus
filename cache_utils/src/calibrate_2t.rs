use crate::calibration::{CalibrateResult, CalibrateResult2T, HashMap, ASVP};

pub fn calibration_result_to_ASVP<T, Analysis: Fn(CalibrateResult) -> T>(
    results: Vec<CalibrateResult2T>,
    analysis: Analysis,
    slicing: &impl Fn(usize) -> u8,
) -> Result<HashMap<ASVP, T>, nix::Error> {
    let mut analysis_result: HashMap<ASVP, T> = HashMap::new();
    for calibrate_2t_result in results {
        let attacker = calibrate_2t_result.main_core;
        let victim = calibrate_2t_result.helper_core;
        match calibrate_2t_result.res {
            Err(e) => return Err(e),
            Ok(calibrate_1t_results) => {
                for result_1t in calibrate_1t_results {
                    let offset = result_1t.offset;
                    let page = result_1t.page;
                    let addr = page + offset as usize;
                    let slice = slicing(addr as usize);
                    let analysed = analysis(result_1t);
                    let asvp = ASVP {
                        attacker,
                        slice,
                        victim,
                        page,
                    };
                    analysis_result.insert(asvp, analysed);
                }
            }
        }
    }
    Ok(analysis_result)
}
