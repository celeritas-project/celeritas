//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedTimeAndRedirect.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

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
 *
 * \warning During scope, you should be sure *NOT* to call the logger, which by
 * default prints to cerr.
 */
class ScopedTimeAndRedirect
{
  public:
    explicit ScopedTimeAndRedirect(std::string label);
    ~ScopedTimeAndRedirect();
    //!@{
    //! no move; no copying
    ScopedTimeAndRedirect(ScopedTimeAndRedirect const&) = delete;
    ScopedTimeAndRedirect& operator=(ScopedTimeAndRedirect const&) = delete;
    ScopedTimeAndRedirect(ScopedTimeAndRedirect&&) = delete;
    ScopedTimeAndRedirect& operator=(ScopedTimeAndRedirect&&) = delete;
    //!@}

  private:
    std::unique_ptr<ScopedStreamRedirect> stdout_;
    std::unique_ptr<ScopedStreamRedirect> stderr_;
    std::string label_;
    ScopedTimeLog scoped_time_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
