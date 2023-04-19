//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleCalo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/Label.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/StreamStore.hh"
#include "corecel/io/OutputInterface.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/user/StepInterface.hh"

#include "SimpleCaloData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition in volumes.
 */
class SimpleCalo final : public StepInterface, public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecLabel = std::vector<Label>;
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    using Energy = units::MevEnergy;
    template<MemSpace M>
    using EnergyRef
        = celeritas::Collection<Energy, Ownership::reference, M, DetectorId>;
    using VecEnergy = std::vector<Energy>;
    //!@}

  public:
    // Construct with sensitive regions
    SimpleCalo(VecLabel labels, SPConstGeo geo, size_type max_streams);

    //!@{
    //! \name Step interface
    // Selection of data required for this interface
    Filters filters() const final;
    // Selection of data required for this interface
    StepSelection selection() const final;
    // Process CPU-generated hits
    void process_steps(HostWTFStepState) final;
    // Process device-generated hits
    void process_steps(DeviceWTFStepState) final;
    //!@}

    //!@{
    //! \name Output interface
    // Category of data to write
    Category category() const final { return Category::result; }
    // Key for the entry inside the category.
    std::string label() const final { return "simple_calo"; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    //// ACCESSORS ////

    //! Number of distinct sensitive volumes
    DetectorId::size_type num_detectors() const { return volume_ids_.size(); }

    // Get tallied stream-local data (throw if not available)
    template<MemSpace M>
    EnergyRef<M> const& energy_deposition(StreamId) const;

    // Get accumulated energy deposition over all streams, on host.
    VecEnergy calc_total_energy_deposition() const;

    //// MUTATORS ////

    // Reset energy deposition to zero, usually at the start of an event
    void reset();

  private:
    VecLabel volume_labels_;
    std::vector<VolumeId> volume_ids_;
    StreamStore<SimpleCaloParamsData, SimpleCaloStateData> store_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
