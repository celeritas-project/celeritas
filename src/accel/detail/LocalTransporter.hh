//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LocalTransporter.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    LocalTransporter ...;
   \endcode
 */
class LocalTransporter
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    // Construct with shared (MT) params
    LocalTransporter(SharedParams, SPCOptions, ...);

    // Convert a Geant4 track to a Celeritas and add to buffer
    void add(const G4Track&);

    // Transport all buffered tracks to completion
    void flush();

  private:
    std::vector<Primary> buffer_;
    std::shared_ptr<StepperInterface> stepper_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
