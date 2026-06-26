use std::fs::File;
use std::io::{self, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};

use rsomics_normaltest::{Alternative, Test, parse_values, run_test};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TestArg {
    Normaltest,
    Skewtest,
    Kurtosistest,
}

impl From<TestArg> for Test {
    fn from(t: TestArg) -> Self {
        match t {
            TestArg::Normaltest => Test::Normaltest,
            TestArg::Skewtest => Test::Skewtest,
            TestArg::Kurtosistest => Test::Kurtosistest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AlternativeArg {
    TwoSided,
    Less,
    Greater,
}

impl From<AlternativeArg> for Alternative {
    fn from(a: AlternativeArg) -> Self {
        match a {
            AlternativeArg::TwoSided => Alternative::TwoSided,
            AlternativeArg::Less => Alternative::Less,
            AlternativeArg::Greater => Alternative::Greater,
        }
    }
}

/// D'Agostino-Pearson normality battery (`scipy.stats.normaltest` and friends).
///
/// Input is one numeric value per line; `-` or omitted reads stdin. `--test`
/// selects the member: `normaltest` (omnibus K² = Z_skew² + Z_kurt², two-sided
/// chi-squared with 2 df), `skewtest` (D'Agostino 1970 skew z-score) or
/// `kurtosistest` (Anscombe-Glynn 1983 kurtosis z-score). Output is a single line
/// `statistic<TAB>p`, where statistic is K² for normaltest or the z-score for the
/// other two. `--alternative` applies to skewtest/kurtosistest only.
#[derive(Parser, Debug)]
#[command(name = "rsomics-normaltest", version, about, long_about = None)]
pub struct Cli {
    /// Input sample (one value per line); `-` or omitted reads stdin.
    #[arg(value_name = "DATA")]
    pub data: Option<PathBuf>,

    /// Which test to run.
    #[arg(long, value_enum, default_value = "normaltest")]
    pub test: TestArg,

    /// Alternative hypothesis (skewtest/kurtosistest only; normaltest is two-sided).
    #[arg(long, value_enum, default_value = "two-sided")]
    pub alternative: AlternativeArg,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let values = match &self.data {
                Some(p) if p.as_os_str() != "-" => {
                    let f = File::open(p).map_err(RsomicsError::Io)?;
                    parse_values(BufReader::new(f))?
                }
                _ => {
                    let stdin = io::stdin();
                    parse_values(stdin.lock())?
                }
            };
            let result = run_test(&values, self.test.into(), self.alternative.into())?;
            if !common.json {
                println!("{}\t{}", result.statistic, result.p);
            }
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
