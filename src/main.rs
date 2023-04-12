use peroxide::fuga::*;

const N: usize = 10000;
const M: usize = 20;
const MTEST: usize = 10000;

fn main() {
    let n = Normal(0.0, 0.1);
    let mut eps = n.sample(N);
    let mut ics = (0usize .. N).collect::<Vec<_>>();
    ics.shuffle(&mut thread_rng());
    let ics_picked = ics[0 .. M].to_vec();
    for i in ics_picked {
        eps[i] *= 3.0;
    }

    let mut eps_test = n.sample(N);
    let mut ics_test = (0usize .. N).collect::<Vec<_>>();
    ics_test.shuffle(&mut thread_rng());
    let ics_picked_test = ics_test[0 .. MTEST].to_vec();
    for i in ics_picked_test {
        eps_test[i] *= 3.0;
    }

    let x = linspace(0, std::f64::consts::PI, N);
    let y = x.fmap(|t| t.sin());
    let y_hat = zip_with(|x, e| x + e, &y, &eps);
    let y_test = zip_with(|x, e| x + e, &y, &eps_test);
    let y_up = y.fmap(|y| y + 3.0 * 0.1);
    let y_down = y.fmap(|y| y - 3.0 * 0.1);

    let l1 = zip_with(|y, y_hat| (y - y_hat).abs(), &y, &y_hat);
    let l1_test = zip_with(|y, y_hat| (y - y_hat).abs(), &y, &y_test);

    // Ordinary 3-sigma rule
    let mean = l1.mean();
    let std = l1.sd();
    println!("Mean: {:.4e}\tStd: {:.4e}", mean, std);
    let upper_bound = mean + 3.0 * std;
    let psqi = l1_test.iter().map(|&x| x < upper_bound).collect::<Vec<_>>();

    // Trimmed 3-sigma rule
    let (mean, std) = iqr_trimmed_mean_std(&l1);
    println!("Mean: {:.4e}\tStd: {:.4e}", mean, std);
    let upper_bound_trimmed = mean + 3.0 * std;
    let psqi_trimmed = l1_test.iter().map(|&x| x < upper_bound_trimmed).collect::<Vec<_>>();

    // Not absolute 3-sigma rule
    let l1_signed = zip_with(|y, y_hat| y_hat - y, &y, &y_hat);
    let mean = l1_signed.mean();
    let std = l1_signed.sd();
    println!("Mean: {:.4e}\tStd: {:.4e}", mean, std);
    let upper_bound_signed = mean + 3.0 * std;
    let psqi_signed = l1_test.iter().map(|&x| x < upper_bound_signed).collect::<Vec<_>>();

    // Not absolute trimmed 3-sigma rule
    let (mean, std) = iqr_trimmed_mean_std(&l1_signed);
    println!("Mean: {:.4e}\tStd: {:.4e}", mean, std);
    let upper_bound_signed_trimmed = mean + 3.0 * std;
    let psqi_signed_trimmed = l1_test.iter().map(|&x| x < upper_bound_signed_trimmed).collect::<Vec<_>>();

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("y", Series::new(y));
    df.push("y_hat", Series::new(y_hat));
    df.push("y_test", Series::new(y_test));
    df.push("y_up", Series::new(y_up));
    df.push("y_down", Series::new(y_down));
    df.push("l1", Series::new(l1));
    df.push("l1_s", Series::new(l1_signed));
    df.push("psqi", Series::new(psqi));
    df.push("psqi_t", Series::new(psqi_trimmed));
    df.push("psqi_s", Series::new(psqi_signed));
    df.push("psqi_st", Series::new(psqi_signed_trimmed));
    df.push("ub", Series::new(vec![upper_bound; N]));
    df.push("ub_t", Series::new(vec![upper_bound_trimmed; N]));
    df.push("ub_s", Series::new(vec![upper_bound_signed; N]));
    df.push("ub_st", Series::new(vec![upper_bound_signed_trimmed; N]));

    df.print();

    df.write_parquet("test.parquet", CompressionOptions::Uncompressed).unwrap();
}

fn iqr_trimmed_mean_std(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    let x = x[n/4 .. 3*n/4].to_vec();
    let mean = x.mean();
    let std = x.sd();
    (mean, std)
}
