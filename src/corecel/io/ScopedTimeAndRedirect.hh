//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedTimeAndRedirect.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Macros.hh"

#include "ScopedStreamRedirect.hh"
#include "ScopedTimeLog.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * At end of scope, print elapsed time and captured cout/cerr.
 *
 * This is designed to prevent other libraries (Geant4,VecGeom) from polluting
 * stdout and breaking JSON reading ability.
 *
 * \code
    {
        ScopedTimeAndRedirect temp_{"VecGeom"};
        vecgeom::DoNoisyAndLongStuff();
    }
   \endcode
 */
class ScopedTimeAndRedirect
{
  public:
    explicit ScopedTimeAndRedirect(std::string label);
    ~ScopedTimeAndRedirect();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedTimeAndRedirect);
    //!@}

  private:
    std::unique_ptr<ScopedStreamRedirect> stdout_;
    std::unique_ptr<ScopedStreamRedirect> stderr_;
    std::string label_;
    ScopedTimeLog scoped_time_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
