//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageWriter.libpng.cc
//---------------------------------------------------------------------------//
#include "ImageWriter.hh"

#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <png.h>

#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
struct FileDeleter
{
    void operator()(std::FILE* ptr)
    {
        auto result = std::fclose(ptr);
        if (result)
        {
            CELER_LOG(error)
                << "Failed to close PNG file: " << std::strerror(errno);
        }
    }
};

void log_png_warning(png_structp, png_const_charp msg)
{
    CELER_LOG(warning) << msg;
}

void log_png_error(png_structp png_ptr, png_const_charp msg)
{
    CELER_LOG(error) << msg;
    std::longjmp(png_jmpbuf(png_ptr), 1);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
struct ImageWriter::Impl
{
    std::unique_ptr<std::FILE, FileDeleter> file{nullptr};
    png_structp png{nullptr};
    png_infop info{nullptr};
};

//---------------------------------------------------------------------------//
//! Deleter for PIMPL idiom
void ImageWriter::ImplDeleter::operator()(Impl* ptr)
{
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a filename and dimensions.
 *
 * Dimensions are row-major like other C++ arrays.
 */
ImageWriter::ImageWriter(std::string const& filename, Size2 height_width)
    : size_{height_width}
{
    CELER_VALIDATE(height_width[0] > 0 && height_width[1] > 0,
                   << "invalid image dimensions: " << height_width[1]
                   << " (W) x " << height_width[0] << " (H)");
    // Create file before making impl in case file output goes wrong
    {
        std::FILE* f = std::fopen(filename.c_str(), "wb");
        CELER_VALIDATE(
            f, << "failed to open output file at '" << filename << '\'');
        impl_.reset(new Impl);
        impl_->file.reset(f);
    }

    // Create output struct
    impl_->png = png_create_write_struct(
        PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (setjmp(png_jmpbuf(impl_->png)))
    {
        CELER_RUNTIME_THROW("libpng", "Failed initialization", {});
    }

    // Set up a warning callback
    png_set_error_fn(impl_->png, nullptr, log_png_error, log_png_warning);

    // Set the output handle
    png_init_io(impl_->png, impl_->file.get());

    // Write header (8 bit colour depth)
    impl_->info = png_create_info_struct(impl_->png);
    png_set_IHDR(impl_->png,
                 impl_->info,
                 height_width[1],
                 height_width[0],
                 sizeof(Color::byte_type) * 8,
                 PNG_COLOR_TYPE_RGB_ALPHA,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    // Create metadata
    {
        static char const software_str[] = "libpng (via Celeritas)";
        png_text text[] = {{}};
        text[0].compression = PNG_TEXT_COMPRESSION_NONE;
        text[0].key = const_cast<char*>("Software");
        text[0].text = const_cast<char*>(software_str);
        text[0].text_length = std::size(software_str);
        png_set_text(impl_->png, impl_->info, text, std::size(text));
    }

    // Save info
    png_write_info(impl_->png, impl_->info);
}

//---------------------------------------------------------------------------//
/*!
 * Write a row.
 */
void ImageWriter::operator()(Span<Color const> colors)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(rows_written_ < size_[0]);
    CELER_EXPECT(colors.size() == size_[1]);

    if (setjmp(png_jmpbuf(impl_->png)))
    {
        CELER_RUNTIME_THROW("libpng", "Failed to write line", {});
    }

    png_write_row(
        impl_->png,
        reinterpret_cast<png_byte*>(const_cast<Color*>(colors.data())));

    ++rows_written_;
}

//---------------------------------------------------------------------------//
/*!
 * Close the file early so that exceptions can be caught.
 */
void ImageWriter::close_impl(bool validate)
{
    if (impl_)
    {
        bool failed{false};

        if (!setjmp(png_jmpbuf(impl_->png)))
        {
            if (rows_written_ != size_[0])
            {
                CELER_LOG(error) << "PNG file received only " << rows_written_
                                 << " of " << size_[0] << " lines";
                png_write_flush(impl_->png);
            }
            png_write_end(impl_->png, impl_->info);
        }
        else
        {
            failed = true;
        }

        png_destroy_write_struct(&impl_->png, &impl_->info);
        impl_.reset();
        if (failed)
        {
            if (validate)
            {
                CELER_RUNTIME_THROW("libpng", "Failed to write file", {});
            }
            CELER_LOG(error) << "libpng failed to write file";
        }
    }
    CELER_ENSURE(!(*this));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
