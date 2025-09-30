//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedStreamRedirect.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <sstream>
#include <string>

#include "corecel/Macros.hh"

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
 *
 * The environment variable \c CELER_DISABLE_REDIRECT will prevent stream
 * redirection, which might be needed if the code segfaults/aborts before this
 * class's destructor is reached.
 */
class ScopedStreamRedirect
{
  public:
    // Whether stream redirection is enabled
    static bool enabled();

    // Construct with pointer to a stream e.g. cout
    explicit ScopedStreamRedirect(std::ostream* os);

    // Restore stream on destruction
    ~ScopedStreamRedirect();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedStreamRedirect);
    //!@}

    // Get redirected output, with trailing whitespaces removed
    std::string str();

    // Get the raw stream after flushing the input
    std::stringstream& get();

  private:
    // >>> DATA

    // Holds a reference to the stream being redirected
    std::ostream* input_stream_;

    // Stores the redirected streams output buffer
    std::streambuf* input_buffer_{nullptr};

    // Holds an output buffer to share with the redirected stream
    std::stringstream temp_stream_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
