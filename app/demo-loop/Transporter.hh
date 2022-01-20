//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Transporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/Assert.hh"
#include "base/CollectionStateStore.hh"
#include "base/Types.hh"
#include "geometry/GeoParams.hh"
#include "sim/TrackData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class MaterialParams;
class GeoMaterialParams;
class ParticleParams;
class CutoffParams;
class PhysicsParams;
class AtomicRelaxationParams;
class RngParams;
class TrackInitParams;

//---------------------------------------------------------------------------//
//! Input parameters to the transporter.
struct TransporterInput
{
    //! Arbitrarily high number for not stopping the simulation short
    static constexpr size_type no_max_steps()
    {
        return celeritas::numeric_limits<size_type>::max();
    }

    // Geometry and materials
    std::shared_ptr<const GeoParams>         geometry;
    std::shared_ptr<const MaterialParams>    materials;
    std::shared_ptr<const GeoMaterialParams> geo_mats;

    // Physics
    std::shared_ptr<const ParticleParams>         particles;
    std::shared_ptr<const CutoffParams>           cutoffs;
    std::shared_ptr<const PhysicsParams>          physics;
    std::shared_ptr<const AtomicRelaxationParams> relaxation;

    // Random
    std::shared_ptr<const RngParams> rng;

    // Constants
    size_type max_num_tracks{};
    size_type max_steps{};
    real_type secondary_stack_factor{};
    bool      enable_diagnostics{true};
    bool      sync{};

    //! True if all params are assigned
    explicit operator bool() const
    {
        return geometry && materials && geo_mats && particles && cutoffs
               && physics && rng;
    }
};

//---------------------------------------------------------------------------//
//! Simulation timing results.
struct TransporterTiming
{
    using VecReal = std::vector<real_type>;

    VecReal   steps;   //!< Real time per step
    real_type total{}; //!< Total simulation time

    // Finer-grained timing information within a step
    real_type initialize_tracks{};
    real_type pre_step{};
    real_type along_and_post_step{};
    real_type launch_models{};
    real_type process_interactions{};
    real_type extend_from_secondaries{};
};

//---------------------------------------------------------------------------//
//! Tallied result and timing from transporting a set of primaries
struct TransporterResult
{
    //!@{
    //! Type aliases
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
    virtual ~TransporterBase() = 0;

    // Transport the input primaries and all secondaries produced
    virtual TransporterResult operator()(const TrackInitParams& primaries) = 0;

    //! Access input parameters (TODO hacky)
    virtual const TransporterInput& input() const = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Transport a set of primaries to completion.
 */
template<MemSpace M>
class Transporter : public TransporterBase
{
  public:
    // Construct from parameters
    explicit Transporter(TransporterInput inp);

    // Transport the input primaries and all secondaries produced
    TransporterResult operator()(const TrackInitParams& primaries) final;

    //! Access input parameters (TODO hacky)
    const TransporterInput& input() const final { return input_; }

  private:
    TransporterInput                          input_;
    ParamsData<Ownership::const_reference, M> params_;
    CollectionStateStore<StateData, M>        states_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
