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
//! Instead of deleting a managed FILE pointer, close it
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

//---------------------------------------------------------------------------//
//! Log a warning when libpng emits one
void log_png_warning(png_structp, png_const_charp msg)
{
    CELER_LOG(warning) << msg;
}

//---------------------------------------------------------------------------//
//! Log an error and jump to handler when libpng fails
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

    // Set up warning/error callbacks
    png_set_error_fn(impl_->png, nullptr, log_png_error, log_png_warning);

    // Set the output handle
    png_init_io(impl_->png, impl_->file.get());

    // Write header (8 bit color depth with alpha)
    impl_->info = png_create_info_struct(impl_->png);
    png_set_IHDR(impl_->png,
                 impl_->info,
                 height_width[1],
                 height_width[0],
                 sizeof(Color::byte_type) * 8,
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    row_buffer_.resize(3 * height_width[1]);

    // Create metadata
    {
        static char software_key[] = "Software";
        static char software_str[] = "libpng (via Celeritas)";
        png_text text[] = {{}};
        text[0].compression = PNG_TEXT_COMPRESSION_NONE;
        text[0].key = static_cast<char*>(software_key);
        text[0].text = static_cast<char*>(software_str);
    }

    // Save info
    png_write_info(impl_->png, impl_->info);
}

//---------------------------------------------------------------------------//
/*!
 * Close on destruction.
 */
ImageWriter::~ImageWriter()
{
    if (*this)
    {
        this->close_impl(/* validate = */ false);
    }
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

    auto iter = row_buffer_.begin();
    for (Color c : colors)
    {
        *iter++ = c.channel(Color::Channel::red);
        *iter++ = c.channel(Color::Channel::green);
        *iter++ = c.channel(Color::Channel::blue);
    }

    png_write_row(impl_->png, reinterpret_cast<png_byte*>(row_buffer_.data()));

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
