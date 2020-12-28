use super::{CreationError, Histogram};

#[path = "helpers.rs"]
pub mod helpers;
#[path = "index_calculation.rs"]
mod index_calculation;
#[path = "init.rs"]
mod init;
#[path = "subtract.rs"]
mod subtract;
#[path = "value_calculation.rs"]
mod value_calculation;

#[test]
fn new_err_high_not_double_low() {
    let res = Histogram::<u64>::new_with_bounds(10, 15, 0);
    assert_eq!(CreationError::HighLessThanTwiceLow, res.unwrap_err());
}

#[test]
fn correct_original_min() {
    // until we get const fns, make sure workaround is correct
    assert_eq!(u64::max_value(), super::ORIGINAL_MIN);
}


#[test]
fn shift_lowest_bucket() {
    // TODO shift left does not seem to work, counts vec differs and normalizing_index_offset(last field), at least the
    // is also differing in the java implementation.
    // until we get const fns, make sure workaround is correct
    for shift_amount in (1..2) {
        let mut h1: Histogram<u64> = Histogram::new_with_max(3600*1000*1000, 3).unwrap();
        h1.reset();
        h1.record_n(0, 500);
        h1.record(2);
        h1.record(4);
        h1.record(5);
        h1.record(511);
        h1.record(512);
        h1.record(1023);
        h1.record(1024);
        h1.record(1025);

        let mut h2: Histogram<u64> = h1.clone();
        h2.reset();
        h2.record_n(0, 500);
        h2.record(2 << shift_amount);
        h2.record(4 << shift_amount);
        h2.record(5 << shift_amount);
        h2.record(511 << shift_amount);
        h2.record(512 << shift_amount);
        h2.record(1023 << shift_amount);
        h2.record(1024 << shift_amount);
        h2.record(1025 << shift_amount);
        
        h1.shift_values_left(shift_amount, 1.0);
        assert_eq!(h1.max(), h2.max());
        assert_eq!(h1.min(), h2.min());
        // TODO min max assert enough?
        // normalizing_index_offset differs here, also in the java impl
        // --> counts should differ? or better their positions in the vec?
        // --> can't check the whole hist for equality
    }
}
    #[test]
    fn shift_non_lowest_bucket() {
        // until we get const fns, make sure workaround is correct
        for shift_amount in (0..10) {
            let mut h1: Histogram<u64> = Histogram::new_with_max(3600*1000*1000, 3).unwrap();
            h1.reset();
            h1.record_n(0, 500);
            h1.record(2 << 10);
            h1.record(4 << 10);
            h1.record(5 << 10);
            h1.record(511 << 10);
            h1.record(512 << 10);
            h1.record(1023 << 10);
            h1.record(1024 << 10);
            h1.record(1025 << 10);
            let orig_hist = h1.clone();
            let mut h2: Histogram<u64> = h1.clone();
            h2.reset();
            h2.record_n(0, 500);
            h2.record((2 << 10) << shift_amount);
            h2.record((4 << 10) <<shift_amount);
            h2.record((5 << 10) <<shift_amount);
            h2.record((511 << 10) << shift_amount);
            h2.record((512 << 10) << shift_amount);
            h2.record((1023 << 10) << shift_amount);
            h2.record((1024 << 10) << shift_amount);
            h2.record((1025 << 10) << shift_amount);
            
            h1.shift_values_left(shift_amount, 1.0);
            assert_eq!(h1.min(), h2.min());
            assert_eq!(h1.max(), h2.max());
            h1.shift_values_right(shift_amount, 1.0);
            assert_eq!(h1, orig_hist);
        }
    }

