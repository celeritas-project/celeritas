//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Transporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"

namespace celeritas
{
struct Primary;
}

namespace demo_loop
{
//---------------------------------------------------------------------------//
template<celeritas::MemSpace M>
class Diagnostic;

//---------------------------------------------------------------------------//
struct EnergyDiagInput
{
    using size_type = celeritas::size_type;
    using real_type = celeritas::real_type;

    char      axis{'z'};
    real_type min{-700};
    real_type max{700};
    size_type num_bins{1024};
};

//---------------------------------------------------------------------------//
//! Input parameters to the transporter.
struct TransporterInput
{
    using size_type  = celeritas::size_type;
    using CoreParams = celeritas::CoreParams;

    //! Arbitrarily high number for not stopping the simulation short
    static constexpr size_type no_max_steps()
    {
        return celeritas::numeric_limits<size_type>::max();
    }

    // Stepper input
    std::shared_ptr<const CoreParams> params;
    size_type num_track_slots{}; //!< AKA max_num_tracks
    bool      sync{false};

    // Loop control
    size_type max_steps{};

    // Diagnostic setup
    bool            enable_diagnostics{true};
    EnergyDiagInput energy_diag;

    //! True if all params are assigned
    explicit operator bool() const
    {
        return params && num_track_slots > 0 && max_steps > 0;
    }
};

//---------------------------------------------------------------------------//
//! Simulation timing results.
struct TransporterTiming
{
    using real_type  = celeritas::real_type;
    using VecReal    = std::vector<real_type>;
    using MapStrReal = std::unordered_map<std::string, real_type>;

    VecReal   steps;   //!< Real time per step
    real_type total{}; //!< Total simulation time
    real_type setup{}; //!< One-time initialization cost
    MapStrReal actions{}; //!< Accumulated action timing
};

//---------------------------------------------------------------------------//
//! Tallied result and timing from transporting a set of primaries
struct TransporterResult
{
    //!@{
    //! Type aliases
    using real_type         = celeritas::real_type;
    using size_type         = celeritas::size_type;
    using VecCount          = std::vector<size_type>;
    using VecReal           = std::vector<real_type>;
    using MapStringCount    = std::unordered_map<std::string, size_type>;
    using MapStringVecCount = std::unordered_map<std::string, VecCount>;
    //!@}

    //// DATA ////

    VecCount          initializers; //!< Num starting track initializers
    VecCount          active;       //!< Num tracks active at beginning of step
    VecCount          alive;        //!< Num living tracks at end of step
    VecReal           edep;         //!< Energy deposition along the grid
    MapStringCount    process;      //!< Count of particle/process interactions
    MapStringVecCount steps;        //!< Distribution of steps
    TransporterTiming time;         //!< Timing information
};

//---------------------------------------------------------------------------//
//! Hack: help adapt demo-loop diagnostics to Transporter/Action
struct DiagnosticStore
{
    using MemSpace = celeritas::MemSpace;
    template<MemSpace M>
    using VecUPDiag = std::vector<std::unique_ptr<Diagnostic<M>>>;

    VecUPDiag<MemSpace::host>   host;
    VecUPDiag<MemSpace::device> device;
};

//---------------------------------------------------------------------------//
/*!
 * Interface class for transporting a set of primaries to completion.
 *
 * We might want to change this so that the transport result gets accumulated
 * over multiple calls rather than combining for a single operation, so
 * diagnostics would be an acessor and the "call" operator would be renamed
 * "transport". Such a change would imply making the diagnostics part of the
 * input parameters, which (for simplicity) isn't done yet.
 */
class TransporterBase
{
  public:
    //!@{
    //! Type aliases
    using VecPrimary = std::vector<celeritas::Primary>;
    using CoreParams = celeritas::CoreParams;
    using ActionId   = celeritas::ActionId;
    //!@}

  public:
    virtual ~TransporterBase() = 0;

    // Transport the input primaries and all secondaries produced
    virtual TransporterResult operator()(const VecPrimary& primaries) = 0;

    //! Access input parameters (TODO hacky)
    const CoreParams& params() const { return *input_.params; }

  protected:
    // TODO: these protected data are a hack for now
    TransporterInput input_;
};

//---------------------------------------------------------------------------//
/*!
 * Transport a set of primaries to completion.
 */
template<celeritas::MemSpace M>
class Transporter final : public TransporterBase
{
  public:
    // Construct from parameters
    explicit Transporter(TransporterInput inp);

    // Transport the input primaries and all secondaries produced
    TransporterResult operator()(const VecPrimary& primaries) final;

  private:
    std::shared_ptr<DiagnosticStore> diagnostics_;
    celeritas::ActionId              diagnostic_action_;
};

//---------------------------------------------------------------------------//
} // namespace demo_loop
