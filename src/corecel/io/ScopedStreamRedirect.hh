//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedStreamRedirect.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <sstream>
#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Redirect the given stream to an internal stringstream.
 *
 * This is primarily for interfacing with poorly-behaved external libraries
 * that write to cout/cerr by default.
 *
 * \code
    ScopedStreamRedirect silenced(&std::cout);
    LoadVecGeom();
    CELER_LOG(diagnostic) << "Vecgeom said: " << silenced.str();
   \endcode
 */
class ScopedStreamRedirect
{
  public:
    // Construct with pointer to a stream e.g. cout
    explicit ScopedStreamRedirect(std::ostream* os);

    // Restore stream on destruction
    ~ScopedStreamRedirect();

    // Get redirected output, with trailing whitespaces removed
    std::string str();

  private:
    // >>> DATA

    // Holds a reference to the stream being redirected
    std::ostream* input_stream_;

    // Stores the redirected streams output buffer
    std::streambuf* input_buffer_;

    // Holds an output buffer to share with the redirected stream
    std::stringstream temp_stream_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
