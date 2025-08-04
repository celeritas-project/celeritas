//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleCalo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/StreamStore.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/OutputInterface.hh"
#include "celeritas/Quantities.hh"

#include "SimpleCaloData.hh"
#include "StepInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeoParamsInterface;

//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition in volumes.
 *
 * \todo Add a "begin run" interface to set up the stream store, rather than
 * passing in number of streams at construction time.
 */
class SimpleCalo final : public StepInterface, public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecLabel = std::vector<Label>;
    using EnergyUnits = units::Mev;
    template<MemSpace M>
    using DetectorRef
        = celeritas::Collection<real_type, Ownership::reference, M, DetectorId>;
    using VecReal = std::vector<real_type>;
    //!@}

  public:
    // Construct with all requirements
    SimpleCalo(std::string output_label,
               VecLabel labels,
               GeoParamsInterface const& geo,
               size_type max_streams);

    //! Construct with default label
    SimpleCalo(VecLabel labels,
               GeoParamsInterface const& geo,
               size_type max_streams)
        : SimpleCalo{"simple_calo", std::move(labels), geo, max_streams}
    {
    }

    //!@{
    //! \name Step interface

    // Map volume names to detector IDs and exclude tracks with no deposition
    Filters filters() const final;
    // Save energy deposition and pre-step volume
    StepSelection selection() const final;
    // Process CPU-generated hits
    void process_steps(HostStepState) final;
    // Process device-generated hits
    void process_steps(DeviceStepState) final;
    //!@}

    //!@{
    //! \name Output interface

    // Category of data to write
    Category category() const final { return Category::result; }
    // Key for the entry inside the category.
    std::string_view label() const final { return output_label_; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    //// ACCESSORS ////

    //! Number of distinct sensitive volumes
    DetectorId::size_type num_detectors() const { return volume_ids_.size(); }

    // Get tallied stream-local data (throw if not available) [EnergyUnits]
    template<MemSpace M>
    DetectorRef<M> const& energy_deposition(StreamId) const;

    // Get accumulated energy deposition over all streams and host/device
    VecReal calc_total_energy_deposition() const;

    //// MUTATORS ////

    // Reset energy deposition to zero, usually at the start of an event
    void clear();

  private:
    using StoreT = StreamStore<SimpleCaloParamsData, SimpleCaloStateData>;

    std::string output_label_;
    VecLabel volume_labels_;
    std::vector<ImplVolumeId> volume_ids_;
    StoreT store_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
