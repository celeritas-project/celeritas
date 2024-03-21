//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OpticalProcess.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    AbsorptionProcess ...;
   \endcode
 */
class AbsorptionProcess : public OpticalProcess
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<AbsorptionParam>;
    //!@}

  public:
    inline CELER_FUNCTION
    AbsorptionProcess(SPConstImported const& shared_data);

    //! Get the interaction cross sections for optical photons
    StepLimitBuilder step_limits() const final;

    //! Apply the interaction kernel on host
    void execute(CoreParams const&, CoreStateHost&) const final;

    //! Apply the interaction kernel on device
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    HostRef const& host_ref() const { return data_.host_ref(); }
    DeviceRef const& device_ref() const { return data_.device_ref(); }
    //!@}

  private:
    SPConstImported data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
