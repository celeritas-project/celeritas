//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimPhotons.h>
#include <lardataobj/Simulation/sim.h>

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "geocel/Types.hh"
#include "celeritas/optical/DetectorData.hh"

#include "detail/LarRunnerDiagnostics.hh"

namespace sim
{
class SimEnergyDeposit;
class OBTRHelper;
}  // namespace sim

namespace celeritas
{
class OutputRegistry;
namespace inp
{
struct OpticalStandaloneInput;
}
namespace optical
{
class Runner;
}  // namespace optical

//---------------------------------------------------------------------------//
/*!
 * Setup and run a standalone optical simulation.
 *
 * This class manages the interface between LArSoft data objects and Celeritas.
 * It is separated from the \c PDFullSimCeler plugin to allow testing
 * and extension to future plugin frameworks (e.g., Phlex).
 * Instantiating the class sets up Celeritas shared and state objects using an
 * input configuration, and each call take a set of energy deposition steps and
 * returns a vector of detector hits.
 *
 * The implementation of this class sets up a standalone Celeritas optical
 * simulation using internal code to extract hits and "backtracker"
 * metadata. Conversion between Celeritas objects and the LArSoft data model
 * happens in this class inside the "call" operator.
 *
 * Since LArSoft is single-threaded, this runner uses only a single "stream".
 * We can in theory enable OpenMP to support parallelism across multiple CPUs
 * in a single-process execution. The class is also \em stateful because it
 * stores the backtracker record helpers and particle metadata.

 * Requirements for PDFastSimPar run:
 * - IncludePropTime: true
 * - UseLitePhotons: true
 * - GeoPropTimeOnly: false
 * Differences:
 * - Reflected and unreflected light are combined
 * TODO:
 * - OnlyActiveVolume: check ISTPC::isScintInActiveVolume
 *
 * \par Construction
 * See \c celeritas::inp::OpticalStandaloneInput .
 */
class LarStandaloneRunner
{
  public:
    //!@{
    //! \name Type aliases
    using VecSED = std::vector<sim::SimEnergyDeposit>;
    using VecBTR = std::vector<sim::OpDetBacktrackerRecord>;
    using VecSPL = std::vector<sim::SimPhotonsLite>;
    using Input = inp::OpticalStandaloneInput;
    using VecReal3 = std::vector<Real3>;
    //!@}

    //! Calculated output from an event
    struct result_type
    {
        VecSPL sim_photons;
        VecBTR backtrack;
    };

  public:
    // Set up the problem, including detector ID coordinates
    LarStandaloneRunner(Input&&, VecReal3 const& det_coords);
    // Don't allow copies of this class
    CELER_DEFAULT_MOVE_DELETE_COPY(LarStandaloneRunner);

    // Run optical photons from a single set of energy steps
    result_type operator()(VecSED const& edep);

  private:
    //// TYPES ////
    using SpanCelerHits = Span<optical::DetectorHit const>;
    using MapIntInt = std::unordered_map<int, int>;

    struct StepMetadata
    {
        //! Unique LArG4 track ID
        //! see ParticleListActionService::preUserTrackingAction
        int track_id = sim::NoParticleId;
        //! Energy deposit per emitted photon
        double avg_edep{};
        //! Midpoint of step (nativeLArSoft units)
        Array<double, 3> midpoint{};
    };

    //// DATA ////

    //!@{
    //! \name Problem setup

    std::shared_ptr<optical::Runner> runner_;
    // Celeritas volume instance ID for each LArSoft detector channel
    std::unordered_map<VolumeInstanceId, unsigned int> geo_to_channel_;
    // Diagnostics written at every step
    std::shared_ptr<OutputRegistry> output_;
    std::shared_ptr<detail::LarRunnerDiagnostics> diagnostics_;

    //!@}
    //!@{
    //! \name Temporary state

    // Energy deposits
    std::vector<StepMetadata> step_md_;
    // Hit recorders for each celeritas volume instance ID
    std::vector<std::unique_ptr<sim::OBTRHelper>> btr_helpers_;
    //! Photon hit count, by detector and rounded ns
    std::vector<MapIntInt> lite_hits_;

    //!@}

    //// HELPERS ////

    std::size_t num_channels() const { return geo_to_channel_.size(); }
    void hit(SpanCelerHits);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
