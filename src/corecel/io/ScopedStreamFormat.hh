//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedStreamFormat.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ios>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Save a stream's state and restore on destruction.
 *
 * Example:
 * \code
     {
         ScopedStreamFormat save_fmt(&std::cout);
         std::cout << setprecision(16) << 1.0;
     }
 * \endcode
 */
class ScopedStreamFormat
{
  public:
    // Construct with stream to safe
    explicit inline ScopedStreamFormat(std::ios* s);

    // Restore formats on destruction
    inline ~ScopedStreamFormat();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedStreamFormat)
    //!@}

  private:
    std::ios* stream_;
    std::ios orig_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
ScopedStreamFormat::ScopedStreamFormat(std::ios* s)
    : stream_{s}, orig_{nullptr}
{
    CELER_EXPECT(s);
    orig_.copyfmt(*s);
}

//---------------------------------------------------------------------------//
/*!
 * Restore formats on destruction.
 */
ScopedStreamFormat::~ScopedStreamFormat()
{
    stream_->copyfmt(orig_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
