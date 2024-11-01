import functools
import operator
import re

from PIL import Image

#
# helpers


def _border(border):
    if isinstance(border, tuple):
        if len(border) == 2:
            left, top = right, bottom = border
        elif len(border) == 4:
            left, top, right, bottom = border
    else:
        left = top = right = bottom = border
    return left, top, right, bottom


def _color(color, mode):
    if isinstance(color, str):
        from . import ImageColor

        color = ImageColor.getcolor(color, mode)
    return color


def _lut(image, lut):
    if image.mode == "P":
        # FIXME: apply to lookup table, not image data
        raise NotImplementedError("mode P support coming soon")
    elif image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        raise OSError("not supported for this image mode")


#
# actions


def autocontrast(images, cutoff=0, ignore=None, mask=None, preserve_tone=False):
    # 自动对比度
    image = images[0]

    if preserve_tone:
        histogram = image.convert("L").histogram(mask)
    else:
        histogram = image.histogram(mask)

    lut = []
    for layer in range(0, len(histogram), 256):
        h = histogram[layer : layer + 256]
        if ignore is not None:
            # get rid of outliers
            try:
                h[ignore] = 0
            except TypeError:
                # assume sequence
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            # cut off pixels from both ends of the histogram
            if not isinstance(cutoff, tuple):
                cutoff = (cutoff, cutoff)
            # get number of pixels
            n = 0
            for ix in range(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = n * cutoff[0] // 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the high end
            cut = n * cutoff[1] // 100
            for hi in range(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        # find lowest/highest samples after preprocessing
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            lut.extend(list(range(256)))
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                ix = int(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut.append(ix)
    my_image = _lut(image, lut)
    trans = [my_image, images[1]]
    return trans


def colorize(image, black, white, mid=None, blackpoint=0, whitepoint=255, midpoint=127):
    """
    Colorize grayscale image.
    This function calculates a color wedge which maps all black pixels in
    the source image to the first color and all white pixels to the
    second color. If ``mid`` is specified, it uses three-color mapping.
    The ``black`` and ``white`` arguments should be RGB tuples or color names;
    optionally you can use three-color mapping by also specifying ``mid``.
    Mapping positions for any of the colors can be specified
    (e.g. ``blackpoint``), where these parameters are the integer
    value corresponding to where the corresponding color should be mapped.
    These parameters must have logical order, such that
    ``blackpoint <= midpoint <= whitepoint`` (if ``mid`` is specified).

    :param image: The image to colorize.
    :param black: The color to use for black input pixels.
    :param white: The color to use for white input pixels.
    :param mid: The color to use for midtone input pixels.
    :param blackpoint: an int value [0, 255] for the black mapping.
    :param whitepoint: an int value [0, 255] for the white mapping.
    :param midpoint: an int value [0, 255] for the midtone mapping.
    :return: An image.
    """

    # Initial asserts
    assert image.mode == "L"
    if mid is None:
        assert 0 <= blackpoint <= whitepoint <= 255
    else:
        assert 0 <= blackpoint <= midpoint <= whitepoint <= 255

    # Define colors from arguments
    black = _color(black, "RGB")
    white = _color(white, "RGB")
    if mid is not None:
        mid = _color(mid, "RGB")

    # Empty lists for the mapping
    red = []
    green = []
    blue = []

    # Create the low-end values
    for i in range(0, blackpoint):
        red.append(black[0])
        green.append(black[1])
        blue.append(black[2])

    # Create the mapping (2-color)
    if mid is None:

        range_map = range(0, whitepoint - blackpoint)

        for i in range_map:
            red.append(black[0] + i * (white[0] - black[0]) // len(range_map))
            green.append(black[1] + i * (white[1] - black[1]) // len(range_map))
            blue.append(black[2] + i * (white[2] - black[2]) // len(range_map))

    # Create the mapping (3-color)
    else:

        range_map1 = range(0, midpoint - blackpoint)
        range_map2 = range(0, whitepoint - midpoint)

        for i in range_map1:
            red.append(black[0] + i * (mid[0] - black[0]) // len(range_map1))
            green.append(black[1] + i * (mid[1] - black[1]) // len(range_map1))
            blue.append(black[2] + i * (mid[2] - black[2]) // len(range_map1))
        for i in range_map2:
            red.append(mid[0] + i * (white[0] - mid[0]) // len(range_map2))
            green.append(mid[1] + i * (white[1] - mid[1]) // len(range_map2))
            blue.append(mid[2] + i * (white[2] - mid[2]) // len(range_map2))

    # Create the high-end values
    for i in range(0, 256 - whitepoint):
        red.append(white[0])
        green.append(white[1])
        blue.append(white[2])

    # Return converted image
    image = image.convert("RGB")
    return _lut(image, red + green + blue)


def contain(image, size, method=Image.BICUBIC):
    """
    Returns a resized version of the image, set to the maximum width and height
    within the requested size, while maintaining the original aspect ratio.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`PIL.Image.BICUBIC`. See :ref:`concept-filters`.
    :return: An image.
    """

    im_ratio = image.width / image.height
    dest_ratio = size[0] / size[1]

    if im_ratio != dest_ratio:
        if im_ratio > dest_ratio:
            new_height = int(image.height / image.width * size[0])
            if new_height != size[1]:
                size = (size[0], new_height)
        else:
            new_width = int(image.width / image.height * size[1])
            if new_width != size[0]:
                size = (new_width, size[1])
    return image.resize(size, resample=method)


def pad(image, size, method=Image.BICUBIC, color=None, centering=(0.5, 0.5)):
    """
    Returns a resized and padded version of the image, expanded to fill the
    requested aspect ratio and size.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`PIL.Image.BICUBIC`. See :ref:`concept-filters`.
    :param color: The background color of the padded image.
    :param centering: Control the position of the original image within the
                      padded version.

                          (0.5, 0.5) will keep the image centered
                          (0, 0) will keep the image aligned to the top left
                          (1, 1) will keep the image aligned to the bottom
                          right
    :return: An image.
    """

    resized = contain(image, size, method)
    if resized.size == size:
        out = resized
    else:
        out = Image.new(image.mode, size, color)
        if resized.width != size[0]:
            x = int((size[0] - resized.width) * max(0, min(centering[0], 1)))
            out.paste(resized, (x, 0))
        else:
            y = int((size[1] - resized.height) * max(0, min(centering[1], 1)))
            out.paste(resized, (0, y))
    return out


def crop(image, border=0):
    """
    Remove border from image.  The same amount of pixels are removed
    from all four sides.  This function works on all image modes.

    .. seealso:: :py:meth:`~PIL.Image.Image.crop`

    :param image: The image to crop.
    :param border: The number of pixels to remove.
    :return: An image.
    """
    left, top, right, bottom = _border(border)
    return image.crop((left, top, image.size[0] - right, image.size[1] - bottom))


def scale(image, factor, resample=Image.BICUBIC):
    """
    Returns a rescaled image by a specific factor given in parameter.
    A factor greater than 1 expands the image, between 0 and 1 contracts the
    image.

    :param image: The image to rescale.
    :param factor: The expansion factor, as a float.
    :param resample: Resampling method to use. Default is
                     :py:attr:`PIL.Image.BICUBIC`. See :ref:`concept-filters`.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    if factor == 1:
        return image.copy()
    elif factor <= 0:
        raise ValueError("the factor must be greater than 0")
    else:
        size = (round(factor * image.width), round(factor * image.height))
        return image.resize(size, resample)


def deform(image, deformer, resample=Image.BILINEAR):
    """
    Deform the image.

    :param image: The image to deform.
    :param deformer: A deformer object.  Any object that implements a
                    ``getmesh`` method can be used.
    :param resample: An optional resampling filter. Same values possible as
       in the PIL.Image.transform function.
    :return: An image.
    """
    return image.transform(image.size, Image.MESH, deformer.getmesh(image), resample)


def equalize(images, mask=None):
    trans = []
    for image in images:
        if image.mode == "P":
            image = image.convert("RGB")
        h = image.histogram(mask)
        lut = []
        for b in range(0, len(h), 256):
            histo = [_f for _f in h[b : b + 256] if _f]
            if len(histo) <= 1:
                lut.extend(list(range(256)))
            else:
                step = (functools.reduce(operator.add, histo) - histo[-1]) // 255
                if not step:
                    lut.extend(list(range(256)))
                else:
                    n = step // 2
                    for i in range(256):
                        lut.append(n // step)
                        n = n + h[i + b]
        trans.append(_lut(image, lut))
    return trans


def expand(image, border=0, fill=0):
    """
    Add border to the image

    :param image: The image to expand.
    :param border: Border width, in pixels.
    :param fill: Pixel fill value (a color value).  Default is 0 (black).
    :return: An image.
    """
    left, top, right, bottom = _border(border)
    width = left + image.size[0] + right
    height = top + image.size[1] + bottom
    color = _color(fill, image.mode)
    if image.mode == "P" and image.palette:
        image.load()
        palette = image.palette.copy()
        if isinstance(color, tuple):
            color = palette.getcolor(color)
    else:
        palette = None
    out = Image.new(image.mode, (width, height), color)
    if palette:
        out.putpalette(palette.palette)
    out.paste(image, (left, top))
    return out


def fit(image, size, method=Image.BICUBIC, bleed=0.0, centering=(0.5, 0.5)):
    """
    Returns a resized and cropped version of the image, cropped to the
    requested aspect ratio and size.

    This function was contributed by Kevin Cazabon.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`PIL.Image.BICUBIC`. See :ref:`concept-filters`.
    :param bleed: Remove a border around the outside of the image from all
                  four edges. The value is a decimal percentage (use 0.01 for
                  one percent). The default value is 0 (no border).
                  Cannot be greater than or equal to 0.5.
    :param centering: Control the cropping position.  Use (0.5, 0.5) for
                      center cropping (e.g. if cropping the width, take 50% off
                      of the left side, and therefore 50% off the right side).
                      (0.0, 0.0) will crop from the top left corner (i.e. if
                      cropping the width, take all of the crop off of the right
                      side, and if cropping the height, take all of it off the
                      bottom).  (1.0, 0.0) will crop from the bottom left
                      corner, etc. (i.e. if cropping the width, take all of the
                      crop off the left side, and if cropping the height take
                      none from the top, and therefore all off the bottom).
    :return: An image.
    """

    # by Kevin Cazabon, Feb 17/2000
    # kevin@cazabon.com
    # http://www.cazabon.com

    # ensure centering is mutable
    centering = list(centering)

    if not 0.0 <= centering[0] <= 1.0:
        centering[0] = 0.5
    if not 0.0 <= centering[1] <= 1.0:
        centering[1] = 0.5

    if not 0.0 <= bleed < 0.5:
        bleed = 0.0

    # calculate the area to use for resizing and cropping, subtracting
    # the 'bleed' around the edges

    # number of pixels to trim off on Top and Bottom, Left and Right
    bleed_pixels = (bleed * image.size[0], bleed * image.size[1])

    live_size = (
        image.size[0] - bleed_pixels[0] * 2,
        image.size[1] - bleed_pixels[1] * 2,
    )

    # calculate the aspect ratio of the live_size
    live_size_ratio = live_size[0] / live_size[1]

    # calculate the aspect ratio of the output image
    output_ratio = size[0] / size[1]

    # figure out if the sides or top/bottom will be cropped off
    if live_size_ratio == output_ratio:
        # live_size is already the needed ratio
        crop_width = live_size[0]
        crop_height = live_size[1]
    elif live_size_ratio >= output_ratio:
        # live_size is wider than what's needed, crop the sides
        crop_width = output_ratio * live_size[1]
        crop_height = live_size[1]
    else:
        # live_size is taller than what's needed, crop the top and bottom
        crop_width = live_size[0]
        crop_height = live_size[0] / output_ratio

    # make the crop
    crop_left = bleed_pixels[0] + (live_size[0] - crop_width) * centering[0]
    crop_top = bleed_pixels[1] + (live_size[1] - crop_height) * centering[1]

    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)

    # resize the image and return it
    return image.resize(size, method, box=crop)


def flip(image):
    """
    Flip the image vertically (top to bottom).

    :param image: The image to flip.
    :return: An image.
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def grayscale(image):
    """
    Convert the image to grayscale.

    :param image: The image to convert.
    :return: An image.
    """
    return image.convert("L")


def invert(images):
    trans = []
    for image in images:
        lut = []
        for i in range(256):
            lut.append(255 - i)
        trans.append(_lut(image, lut))
    return trans


def mirror(image):
    """
    Flip image horizontally (left to right).

    :param image: The image to mirror.
    :return: An image.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def posterize(images, bits):
    # 减少色彩深度
    image = images[0]
    lut = []
    mask = ~(2 ** (8 - bits) - 1)
    for i in range(256):
        lut.append(i & mask)
    my_image = _lut(image, lut)
    trans = [my_image, images[1]]
    return trans


def solarize(images, threshold=128):
    # 反转亮度
    image = images[0]
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(i)
        else:
            lut.append(255 - i)
    my_image = _lut(image, lut)
    trans = [my_image, images[1]]
    return trans


def exif_transpose(image):
    """
    If an image has an EXIF Orientation tag, return a new image that is
    transposed accordingly. Otherwise, return a copy of the image.

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112)
    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)
    if method is not None:
        transposed_image = image.transpose(method)
        transposed_exif = transposed_image.getexif()
        if 0x0112 in transposed_exif:
            del transposed_exif[0x0112]
            if "exif" in transposed_image.info:
                transposed_image.info["exif"] = transposed_exif.tobytes()
            elif "Raw profile type exif" in transposed_image.info:
                transposed_image.info[
                    "Raw profile type exif"
                ] = transposed_exif.tobytes().hex()
            elif "XML:com.adobe.xmp" in transposed_image.info:
                transposed_image.info["XML:com.adobe.xmp"] = re.sub(
                    r'tiff:Orientation="([0-9])"',
                    "",
                    transposed_image.info["XML:com.adobe.xmp"],
                )
        return transposed_image
    return image.copy()
