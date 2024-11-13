//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageWriter.cc
//---------------------------------------------------------------------------//
#include "ImageWriter.hh"

#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"


namespace celeritas
{
//---------------------------------------------------------------------------//
ImageWriter::ImageWriter(std::string const&, Size2)
{
    CELER_DISCARD(size_);
    CELER_DISCARD(rows_written_);
    CELER_DISCARD(row_buffer_);
    CELER_NOT_CONFIGURED("PNG");
}
void ImageWriter::operator()(Span<Color const>)
{
    CELER_ASSERT_UNREACHABLE();
}
void ImageWriter::close_impl(bool) {}
void ImageWriter::ImplDeleter::operator()(Impl*) {}
//---------------------------------------------------------------------------//
}  // namespace celeritas
