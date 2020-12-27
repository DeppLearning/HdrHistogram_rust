//! WIP port of DoubleHistogram

use crate::{AdditionError, Histogram, RecordError, SubtractionError, errors, iterators::HistogramIterator};
use crate::errors::CreationError;
use crate::Counter;

use std::borrow::Borrow;

/// Port of Double histogram
#[derive(Debug)]
pub struct DoubleHistogram<C: Counter> {
    // A value that will keep us from multiplying into infinity.
    auto_resize: bool,
    highest_allowed_value_ever: f64,
    configured_highest_to_lowest_value_ratio: u64,
    current_lowest_value_in_auto_range: f64,
    current_highest_value_limit_in_auto_range: f64,
    integer_values_histogram: Histogram<C>,
    /// Temporarily: int to double ratio
    pub integer_to_double_value_conversion_ratio: f64,
    double_to_integer_value_conversion_ratio: f64
}

impl <C: Counter> DoubleHistogram<C> {
    fn new_with_args(highest_allowed_value_ever: f64,
        configured_highest_to_lowest_value_ratio: u64,
        current_lowest_value_in_auto_range: f64,
        current_highest_value_limit_in_auto_range: f64,
        sigfig: u8) -> Result<Self, CreationError> {
            if configured_highest_to_lowest_value_ratio < 2 {
                return Err(CreationError::HighLessThanTwiceLow)
            }

            if configured_highest_to_lowest_value_ratio * 10_u64.pow(sigfig as u32) >= (1_u64 << 61) {
                // TODO make dedicated error variant
                return Err(CreationError::SigFigExceedsMax)
            }
            
            let integer_value_range = Self::derive_integer_value_range(configured_highest_to_lowest_value_ratio, sigfig);
            let integer_values_histogram = Histogram::new_with_bounds(1, integer_value_range, sigfig)?;
            // We want the auto-ranging to tend towards using a value range that will result in using the
            // lower tracked value ranges and leave the higher end empty unless the range is actually used.
            // This is most easily done by making early recordings force-shift the lower value limit to
            // accommodate them (forcing a force-shift for the higher values would achieve the opposite).
            // We will therefore start with a very high value range, and let the recordings autoAdjust
            // downwards from there:
            let initial_lowest_value_in_auto_range: f64 = 2.0_f64.powi(800);
            let mut hist = DoubleHistogram {
                auto_resize: false,
                highest_allowed_value_ever,
                configured_highest_to_lowest_value_ratio,
                current_highest_value_limit_in_auto_range,
                current_lowest_value_in_auto_range,
                integer_values_histogram,
                integer_to_double_value_conversion_ratio: 1.0,
                double_to_integer_value_conversion_ratio: 1.0,
            };
            hist.init(configured_highest_to_lowest_value_ratio, initial_lowest_value_in_auto_range);
            Ok(hist)
        }
    fn init(&mut self, configured_highest_to_lowest_value_ratio: u64, lowest_trackable_unit_value: f64) {
        let internal_highest_to_lowest_value_ratio = Self::derive_internal_highest_to_lowest_value_ratio(configured_highest_to_lowest_value_ratio);
        self.set_trackable_value_range(lowest_trackable_unit_value, lowest_trackable_unit_value * internal_highest_to_lowest_value_ratio as f64)
    }
    fn set_trackable_value_range(&mut self, lowest_value_in_auto_range: f64, highest_value_in_auto_range: f64) {
        self.current_lowest_value_in_auto_range = lowest_value_in_auto_range;
        self.current_highest_value_limit_in_auto_range = highest_value_in_auto_range;
        let integer_to_double_value_conversion_ratio: f64 = lowest_value_in_auto_range / self.get_lowest_tracking_integer_value() as f64;
        self.set_integer_to_double_value_conversation_ratio(integer_to_double_value_conversion_ratio);
    }
    fn set_integer_to_double_value_conversation_ratio(&mut self, integer_to_double_value_conversion_ratio: f64) {
        self.integer_to_double_value_conversion_ratio = integer_to_double_value_conversion_ratio;
        self.double_to_integer_value_conversion_ratio = 1.0 / integer_to_double_value_conversion_ratio;
    }
    fn get_lowest_tracking_integer_value(&self) -> u32 {
        self.integer_values_histogram.sub_bucket_half_count
    }

    fn get_count_at_value(&self, value: f64) -> C {
        self.integer_values_histogram.count_at((value * self.double_to_integer_value_conversion_ratio).trunc() as u64)
    }
    /// new
    pub fn new(sigfig: u8) -> Result<Self, CreationError> {
        // TODO fix this for sigfig > 1
        Self::new_with_max(std::f64::MAX, sigfig)
    }
    /// new with max val
    pub fn new_with_max(max_value: f64, sigfig: u8) -> Result<Self, CreationError> {
        let mut h = Self::new_with_args(max_value / 2.0, 2, 1.0, 2.0, sigfig);
        if let Ok(ref mut h) = h {
            h.integer_values_histogram.auto(true);
        }
        h
    }

    // Internal dynamic range needs to be 1 order of magnitude larger than the containing order of magnitude.
    // e.g. the dynamic range that covers [0.9, 2.1) is 2.33x, which on it's own would require 4x range to
    // cover the contained order of magnitude. But (if 1.0 was a bucket boundary, for example, the range
    // will actually need to cover [0.5..1.0) [1.0..2.0) [2.0..4.0), mapping to an 8x internal dynamic range.
    fn derive_integer_value_range(external_highest_to_lowest_value_ratio: u64, sigfig: u8) -> u64 {
        let internal_highest_to_lowest_value_ratio: u64 = Self::derive_internal_highest_to_lowest_value_ratio(external_highest_to_lowest_value_ratio);
        

        // We cannot use the bottom half of bucket 0 in an integer values histogram to represent double
        // values, because the required precision does not exist there. We therefore need the integer
        // range to be bigger, such that the entire double value range can fit in the upper halves of
        // all buckets. Compute the integer value range that will achieve this:
        let lowest_tracking_integer_value: u64 = (Self::number_of_sub_buckets(sigfig) / 2) as u64;
        let integer_value_range: u64 = lowest_tracking_integer_value * internal_highest_to_lowest_value_ratio;

        internal_highest_to_lowest_value_ratio
    }
    fn derive_internal_highest_to_lowest_value_ratio(external_highest_to_lowest_value_ratio: u64) -> u64 {
        1_u64 << (Self::find_containing_binary_order_of_magnitude_u(external_highest_to_lowest_value_ratio) + 1)

    }
    // smallest power of 2 containing value
    fn find_containing_binary_order_of_magnitude_u(value: u64) -> u8 {
        (64 - value.leading_zeros()) as u8
    }

    // smallest power of 2 containing value
    fn find_containing_binary_order_of_magnitude_f(value: f64) -> u8 {
        let value_u: u64 = value.ceil().trunc() as u64;
        Self::find_containing_binary_order_of_magnitude_u(value_u)
    }
    fn number_of_sub_buckets(sigfig: u8) -> u32 {
        let largest_value_with_single_unit_resolution: u64 = 2 * 10_u64.pow(sigfig as u32);

        // We need to maintain power-of-two subBucketCount (for clean direct indexing) that is large enough to
        // provide unit resolution to at least largestValueWithSingleUnitResolution. So figure out
        // largestValueWithSingleUnitResolution's nearest power-of-two (rounded up), and use that:
        let sub_bucket_count_magnitude: u8 = (largest_value_with_single_unit_resolution.as_f64().ln()/ 2.0_f64.ln()).ceil() as u8;
        let sub_bucket_count: u32 = 2_u32.pow(sub_bucket_count_magnitude as u32);
        return sub_bucket_count;

    }
    fn auto_adjust_range_for_value(&mut self, value: f64) -> Result<(), errors::RecordError> {
        if value == 0.0 {
            return Ok(());
        }
        self.auto_adjust_range_for_value_slow_path(value)
    }
    fn auto_adjust_range_for_value_slow_path(&mut self, value: f64) -> Result<(), errors::RecordError> {
        if value < self.current_lowest_value_in_auto_range {
            if (value < 0.0) {
                return Err(errors::RecordError::ValueOutOfRangeResizeDisabled);
            }
            loop {
                //dbg!("1");
                let shift_amount: u8 =
                self.find_capped_containing_binary_order_of_magnitude(
                        (self.current_lowest_value_in_auto_range / value).ceil() - 1.0);
                self.shift_covered_range_to_the_right(shift_amount)?;

                if value >= self.current_lowest_value_in_auto_range {
                    break;
                }
            }
        } else if value >=self.current_highest_value_limit_in_auto_range {
            if value > self.highest_allowed_value_ever {
                return Err(RecordError::ResizeFailedUsizeTypeTooSmall);
            }
            loop {
                dbg!("2");
                    // If value is an exact whole multiple of currentHighestValueLimitInAutoRange, it "belongs" with
                    // the next level up, as it crosses the limit. With floating point values, the simplest way to
                    // make this shift on exact multiple values happen (but not for any just-smaller-than-exact-multiple
                    // values) is to use a value that is 1 ulp bigger in computing the ratio for the shift amount:
                    let shift_amount: u8 =
                            self.find_capped_containing_binary_order_of_magnitude(
                                    float_extras::f64::nextafter(value, std::f64::MAX).ceil() / self.current_highest_value_limit_in_auto_range - 1.0);
                    self.shift_covered_range_to_the_left(shift_amount)?;
                    if value < self.current_highest_value_limit_in_auto_range {
                        break;
                    }
            }
        }

        Ok(())

    }
    // TODO this has a nested try catch finally in the java code, make sure the finally is handled correctly
    fn shift_covered_range_to_the_left(&mut self, number_of_binary_orders_of_magnitude: u8) -> Result<(), RecordError> {
            // We are going to adjust the tracked range by effectively shifting it to the right
            // (in the integer shift sense).
            //
            // To counter the left shift of the value multipliers, we need to right shift the internal
            // representation such that the newly shifted integer values will continue to return the
            // same double values.
    
            // Initially, new range is the same as current range, to make sure we correctly recover
            // from a shift failure if one happens:
            let mut new_lowest_value_in_auto_range = self.current_lowest_value_in_auto_range;
            let mut new_highest_value_limit_in_auto_range = self.current_highest_value_limit_in_auto_range;
    
            let shift_multiplier = 1.0 * (1_u64 << number_of_binary_orders_of_magnitude) as f64;

            let new_integer_to_double_value_conversion_ratio =
                    self.integer_to_double_value_conversion_ratio * shift_multiplier;

            // First, temporarily change the lowest value in auto-range without changing conversion ratios.
            // This is done to force new values lower than the new expected lowest value to attempt an
            // adjustment (which is synchronized and will wait behind this one). This ensures that we will
            // not end up with any concurrently recorded values that would need to be discarded if the shift
            // fails. If this shift succeeds, the pending adjustment attempt will end up doing nothing.
            self.current_lowest_value_in_auto_range *= shift_multiplier;

            // First shift the values, to give the shift a chance to fail:

            // Shift integer histogram right, decreasing the recorded integer values for current recordings
            // by a factor of (1 << numberOfBinaryOrdersOfMagnitude):

            // (no need to shift any values if all recorded values are at the 0 value level:)
            if self.integer_values_histogram.len() > self.integer_values_histogram.count_at_index(0).unwrap().as_u64() {
                // Apply the shift:
                if self.integer_values_histogram.shift_values_right(number_of_binary_orders_of_magnitude,
                            new_integer_to_double_value_conversion_ratio).is_err() {
                                self.handle_shift_values_exception(number_of_binary_orders_of_magnitude)?;
                                new_lowest_value_in_auto_range /= shift_multiplier;
                            }
            }
            // Shift was successful. Adjust new range to reflect:
            new_lowest_value_in_auto_range *= shift_multiplier;
            new_highest_value_limit_in_auto_range *= shift_multiplier;
            // Set the new range to either the successfully changed one, or the original one:
            self.set_trackable_value_range(new_lowest_value_in_auto_range, new_highest_value_limit_in_auto_range);
            Ok(())
            
        }
    // TODO this has a nested try catch finally in the java code, make sure the finally is handled correctly
    fn shift_covered_range_to_the_right(&mut self, number_of_binary_orders_of_magnitude: u8) -> Result<(), RecordError> {
        // We are going to adjust the tracked range by effectively shifting it to the right
        // (in the integer shift sense).
        //
        // To counter the right shift of the value multipliers, we need to left shift the internal
        // representation such that the newly shifted integer values will continue to return the
        // same double values.

        // Initially, new range is the same as current range, to make sure we correctly recover
        // from a shift failure if one happens:
        let mut new_lowest_value_in_auto_range: f64 = self.current_lowest_value_in_auto_range;
        let mut new_highest_value_limit_in_auto_range: f64 = self.current_highest_value_limit_in_auto_range;

        let shift_multiplier: f64 = 1.0 / ( 1_u64 << number_of_binary_orders_of_magnitude) as f64;

        // First, temporarily change the highest value in auto-range without changing conversion ratios.
        // This is done to force new values higher than the new expected highest value to attempt an
        // adjustment (which is synchronized and will wait behind this one). This ensures that we will
        // not end up with any concurrently recorded values that would need to be discarded if the shift
        // fails. If this shift succeeds, the pending adjustment attempt will end up doing nothing.
        self.current_highest_value_limit_in_auto_range *= shift_multiplier;
        let new_integer_to_double_value_conversion_ratio: f64 =
                self.integer_to_double_value_conversion_ratio * shift_multiplier;
        // First shift the values, to give the shift a chance to fail:
        // Shift integer histogram left, increasing the recorded integer values for current recordings
        // by a factor of (1 << numberOfBinaryOrdersOfMagnitude):
        // (no need to shift any values if all recorded values are at the 0 value level:)
        if self.integer_values_histogram.len() > self.integer_values_histogram.count_at_index(0).unwrap().as_u64() {
            if self.integer_values_histogram.shift_values_left(number_of_binary_orders_of_magnitude, new_integer_to_double_value_conversion_ratio).is_err() {
                self.handle_shift_values_exception(number_of_binary_orders_of_magnitude)?;
                new_highest_value_limit_in_auto_range /= shift_multiplier;
                self.integer_values_histogram.shift_values_left(number_of_binary_orders_of_magnitude, new_integer_to_double_value_conversion_ratio)?;
            }
        }
        new_lowest_value_in_auto_range *= shift_multiplier;
        new_highest_value_limit_in_auto_range *= shift_multiplier;
        self.set_trackable_value_range(new_lowest_value_in_auto_range, new_highest_value_limit_in_auto_range);

        Ok(())
    }
    fn handle_shift_values_exception(&mut self, number_of_binary_orders_of_magnitude: u8) -> Result<(), RecordError> {
        if !self.auto_resize {
            return Err(RecordError::ResizeFailedUsizeTypeTooSmall);
        }

        let highest_trackable_value: u64 = self.integer_values_histogram.highest_trackable_value;
        let current_containing_order_of_magnitude = Self::find_containing_binary_order_of_magnitude_u(highest_trackable_value);
        let new_containing_order_of_magnitude = number_of_binary_orders_of_magnitude * current_containing_order_of_magnitude;
        if new_containing_order_of_magnitude > 63 {
            return Err(RecordError::ResizeFailedUsizeTypeTooSmall);
        }
        let new_highest_trackable_value = (1_u64 << new_containing_order_of_magnitude) - 1;
        self.integer_values_histogram.resize(new_highest_trackable_value).unwrap();
        self.integer_values_histogram.highest_trackable_value = new_highest_trackable_value;
        self.configured_highest_to_lowest_value_ratio <<= number_of_binary_orders_of_magnitude;
        Ok(())
    }
    fn find_capped_containing_binary_order_of_magnitude(&self, value: f64) -> u8 {
        if value > self.configured_highest_to_lowest_value_ratio as f64{
            return (self.configured_highest_to_lowest_value_ratio.as_f64().ln() / 2.0_f64.ln()) as u8
        }
        if value > 2.0_f64.powi(50) {
            return 50;
        }
        Self::find_containing_binary_order_of_magnitude_f(value)
    }

    fn record_converted_double_value(&mut self, value: f64) -> Result<(), RecordError> {
        let integer_value: u64 = (value * self.double_to_integer_value_conversion_ratio) as u64;
        self.integer_values_histogram.record(integer_value)
    }
    fn record_single_value(&mut self, value: f64) -> Result<(), errors::RecordError>  {
        let mut throw_count: u8 = 0;
        loop {
            dbg!("3");
            if (value < self.current_lowest_value_in_auto_range) || (value >= self.current_highest_value_limit_in_auto_range) {
                // Zero is valid and needs no auto-ranging, but also rare enough that we should deal
                // with it on the slow path...
                self.auto_adjust_range_for_value(value);
            }
            if self.record_converted_double_value(value).is_ok() {
                return Ok(());
            }
            throw_count += 1;
            if throw_count > 64 {
                return Err(RecordError::ResizeFailedUsizeTypeTooSmall);
            }
        }
    }
    fn record_converted_double_value_with_count(&mut self, value: f64, count: C) -> Result<(), RecordError> {
        let integer_value: u64 = (value * self.double_to_integer_value_conversion_ratio).trunc() as u64;
        self.integer_values_histogram.record_n(integer_value, count)
    }
    /// record w count
    pub fn record_value_with_count(&mut self, value: f64, count: C) -> Result<(), RecordError> {
        let mut throw_count = 0;
        loop {
            if (value < self.current_lowest_value_in_auto_range) || (value >= self.current_highest_value_limit_in_auto_range) {
                // Zero is valid and needs no auto-ranging, but also rare enough that we should deal
                // with it on the slow path...
                self.auto_adjust_range_for_value(value);
            }

            if self.record_converted_double_value_with_count(value, count).is_ok() {
                return Ok(())
            }
            // A race that would pass the auto-range check above and would still take an AIOOB
            // can only occur due to a value that would have been valid becoming invalid due
            // to a concurrent adjustment operation. Such adjustment operations can happen no
            // more than 64 times in the entire lifetime of the Histogram, which makes it safe
            // to retry with no fear of live-locking.

            throw_count += 1;
            if throw_count > 1 {
                // For the retry check to not detect an out of range attempt after 64 retries
                // should be  theoretically impossible, and would indicate a bug.
                panic!("BUG: Unexpected non-transient AIOOB Exception caused");
            }
        }
    }

    /// Add the contents of another histogram to this one.
    ///
    /// Returns an error if values in the other histogram cannot be stored; see `AdditionError`.
    pub fn add<B: Borrow<DoubleHistogram<C>>>(&mut self, source: B) -> Result<(), AdditionError> {
        let other = source.borrow();
        let array_length = other.integer_values_histogram.counts.len();
        for i in 0..array_length {
            let count = other.integer_values_histogram.count_at_index(i).unwrap();
            if count > C::zero() {
                self.record_value_with_count(other.integer_values_histogram.value_for(i) as f64 * other.integer_to_double_value_conversion_ratio, count).map_err(|e| AdditionError::OtherAddendValueExceedsRange)?;
            }
        }
        Ok(())
        // TODO check if this works, probbaly not 
        //self.integer_values_histogram.add(&source.borrow().integer_values_histogram)
    }
    /// sub
    pub fn subtract<B: Borrow<DoubleHistogram<C>>>(
        &mut self,
        subtrahend: B,
    ) -> Result<(), SubtractionError> {
        // let array_length = other.integer_values_histogram.counts.len();
        // for i in 0..array_length {
        //     let count = other.integer_values_histogram.count_at_index(i).unwrap();
        //     if count > C::zero() {
        //         let value = other.integer_values_histogram.value_for(i) as f64 * other.integer_to_double_value_conversion_ratio;
        //         if self.get_count_at_value(value) < count {
        //             return Err(SubtractionError::SubtrahendCountExceedsMinuendCount)
        //         }
        //         self.integer_values_histogram.
        //         self.record_value_with_count(value, -count);
        //     }
        // }
        // TODO enough? adjust ranges?
        self.integer_values_histogram.subtract(&subtrahend.borrow().integer_values_histogram)
    }
    /// mean
    pub fn mean(&self) -> f64 {
        self.integer_values_histogram.mean() * self.integer_to_double_value_conversion_ratio
    }
    /// stdev
    pub fn stdev(&self) -> f64 {
        self.integer_values_histogram.stdev() * self.integer_to_double_value_conversion_ratio
    }

    /// len
    pub fn len(&self) -> u64 {
        self.integer_values_histogram.len()
    }

    /// iter recorded
    pub fn iter_recorded(&self) -> HistogramIterator<'_, C, crate::iterators::recorded::Iter> {
        self.integer_values_histogram.iter_recorded()
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new() {
        let mut h: DoubleHistogram<u8> = DoubleHistogram::new_with_max(16.0, 1).unwrap();
        h.record_single_value(1.5);
        h.record_single_value(2.5);
        //dbg!(&h);
        dbg!(h.mean(), h.stdev());

        let mut h2: DoubleHistogram<u8> = DoubleHistogram::new_with_max(16.0, 1).unwrap();
        h2.record_single_value(1.7);
        h.subtract(h2);
        dbg!(h.mean());
    }
}