use ndarray::Array2;
use crate::snn::architecture::{SpikingEncoder, STCM, SpikingDecoder};

#[derive(Clone)]
pub struct Gradients {
    pub d_we: Array2<f32>,
    pub d_wr_enc: Array2<f32>,
    pub d_wce: Array2<f32>,
    pub d_wcc: Array2<f32>,
    pub d_wctx: Array2<f32>,
    pub d_wself: Array2<f32>,
    pub d_wy: Array2<f32>,
    pub d_wc: Array2<f32>,
    pub d_wr_dec: Array2<f32>,
}

impl Gradients {
    pub fn zeros(encoder: &SpikingEncoder, stcm: &STCM, decoder: &SpikingDecoder) -> Self {
        Self {
            d_we: Array2::zeros(encoder.w_e.raw_dim()),
            d_wr_enc: Array2::zeros(encoder.w_r.raw_dim()),
            d_wce: Array2::zeros(stcm.w_ce.raw_dim()),
            d_wcc: Array2::zeros(stcm.w_cc.raw_dim()),
            d_wctx: Array2::zeros(stcm.w_ctx.raw_dim()),
            d_wself: Array2::zeros(stcm.w_self.raw_dim()),
            d_wy: Array2::zeros(decoder.w_y.raw_dim()),
            d_wc: Array2::zeros(decoder.w_c.raw_dim()),
            d_wr_dec: Array2::zeros(decoder.w_r.raw_dim()),
        }
    }

    pub fn add(&mut self, other: &Self) {
        self.d_we += &other.d_we;
        self.d_wr_enc += &other.d_wr_enc;
        self.d_wce += &other.d_wce;
        self.d_wcc += &other.d_wcc;
        self.d_wctx += &other.d_wctx;
        self.d_wself += &other.d_wself;
        self.d_wy += &other.d_wy;
        self.d_wc += &other.d_wc;
        self.d_wr_dec += &other.d_wr_dec;
    }
}
