from PIL import Image, ImageFilter, ImageStat


class _Enhance:
    def enhance(self, factor):
        """
        Returns an enhanced image.

        :param factor: A floating point value controlling the enhancement.
                       Factor 1.0 always returns a copy of the original image,
                       lower factors mean less color (brightness, contrast,
                       etc), and higher values more. There are no restrictions
                       on this value.
        :rtype: :py:class:`~PIL.Image.Image`
        """
        trans = [Image.blend(self.degenerate, self.images[0], factor), self.images[1]]
        return trans


class MyColor(_Enhance):
    """Adjust image color balance.

    This class can be used to adjust the colour balance of an image, in
    a manner similar to the controls on a colour TV set. An enhancement
    factor of 0.0 gives a black and white image. A factor of 1.0 gives
    the original image.
    """

    def __init__(self, images):
        self.images = images
        self.intermediate_mode = "L"
        if "A" in images[0].getbands():
            self.intermediate_mode = "LA"

        self.degenerate = images[0].convert(self.intermediate_mode).convert(images[0].mode)


class MyContrast(_Enhance):
    """Adjust image contrast.

    This class can be used to control the contrast of an image, similar
    to the contrast control on a TV set. An enhancement factor of 0.0
    gives a solid grey image. A factor of 1.0 gives the original image.
    """

    def __init__(self, images):
        self.images = images
        mean = int(ImageStat.Stat(images[0].convert("L")).mean[0] + 0.5)
        self.degenerate = Image.new("L", images[0].size, mean).convert(images[0].mode)

        if "A" in images[0].getbands():
            self.degenerate.putalpha(images[0].getchannel("A"))


class MyBrightness(_Enhance):
    """Adjust image brightness.

    This class can be used to control the brightness of an image.  An
    enhancement factor of 0.0 gives a black image. A factor of 1.0 gives the
    original image.
    """

    def __init__(self, images):
        self.images = images
        self.degenerate = Image.new(images[0].mode, images[0].size, 0)

        if "A" in images[0].getbands():
            self.degenerate.putalpha(images[0].getchannel("A"))


class MySharpness(_Enhance):
    """Adjust image sharpness.

    This class can be used to adjust the sharpness of an image. An
    enhancement factor of 0.0 gives a blurred image, a factor of 1.0 gives the
    original image, and a factor of 2.0 gives a sharpened image.
    """

    def __init__(self, images):
        self.images = images
        self.degenerate = images[0].filter(ImageFilter.SMOOTH)

        if "A" in images[0].getbands():
            self.degenerate.putalpha(images[0].getchannel("A"))
