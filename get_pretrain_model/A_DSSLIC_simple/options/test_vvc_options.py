from .test_options import TestOptions


class TestVVCOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument("--pix_fmt", type=str, default="yuv420p10le", help="pixel_fmt for ffmpeg")
        self.parser.add_argument("--bit_depth", type=int, default=10, help="bit depth for ffmpeg")
        self.parser.add_argument("--chromafmt", type=str, default="420", help="corresponding to pix_fmt")
        self.parser.add_argument("--is_comp", type=bool, default=False)
        