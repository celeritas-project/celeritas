//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleTransporter.hh
//! \brief Simple example of running the stepper loop on host or device.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/global/Stepper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Control parameters for the simple transporter
struct SimpleTransporterInput
{
    size_type max_step_iterations{}; //!< Number of outer GPU loops

    //! True if inputs are valid
    explicit operator bool() const { return max_step_iterations > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Interface class for transporting a set of primaries to completion.
 *
 * The "output" of running the primaries should be automatic from whatever
 * actions have been registered with the ActionManager.
 */
class SimpleTransporterInterface
{
  public:
    //!@{
    //! Type aliases
    using VecPrimary = std::vector<Primary>;
    //!@}

  public:
    virtual ~SimpleTransporterInterface() = 0;

    // Transport the input primaries and all secondaries produced
    virtual void operator()(const VecPrimary& primaries) = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Transport primaries to completion in a particular memory space.
 */
template<MemSpace M>
class SimpleTransporter final : public SimpleTransporterInterface
{
  public:
    // Construct with transporter parameters and loop control
    SimpleTransporter(StepperInput stepper_inp, SimpleTransporterInput st_inp);

    // Transport the input primaries and all secondaries produced
    void operator()(const VecPrimary& primaries) final;

  private:
    Stepper<M>             step_;
    SimpleTransporterInput st_input_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
