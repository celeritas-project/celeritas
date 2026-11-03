//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "corecel/data/StreamStore.hh"
#include "corecel/io/OutputInterface.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/inp/Diagnostics.hh"

#include "FieldDiagnosticData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
class VolumeParams;

//---------------------------------------------------------------------------//
/*!
 * Characterize the cost of charged particle propagation in a field.
 *
 * When this diagnostic is present, the field along-step actions launch an
 * instrumented propagation kernel instead of the production one. The
 * instrumented kernel counts, for each track and step, the field propagator
 * substeps and their outcomes, the integrations in each loop of the
 * substepper, and the boundary searches (see \c FieldPropagationCounters ).
 * It writes them to a per-slot record that this action bins at the end of the
 * step into:
 * - per-step histograms of substeps, integrations, first-substep
 *   integrations, and boundary searches for each particle type;
 * - per-track histograms of the same quantities summed over the track's
 *   propagation steps, binned when the track is killed;
 * - totals of all counters (\c FieldTally ) for each logarithmic kinetic
 * energy bin and for each canonical volume where the step started;
 * - totals and maxima for each step iteration, which identify the iterations
 *   whose propagation kernels are slowest;
 * - optionally, a histogram of the ratio of the safety distance at the start
 *   of the step to the step length.
 *
 * The optional "oracle" safety calculation at the start of each step also
 * counts the boundary searches whose search segment lies inside that safety
 * sphere: these could be skipped by a propagator that used the safety.
 *
 * The results are written to the \c field-diagnostic entry in the \c result
 * category of the main Celeritas output, accumulated over all streams and
 * events. Step iteration records are written separately for each stream.
 *
 * \note The instrumentation changes the register usage and memory traffic of
 * the propagation kernel, so profile the uninstrumented kernel for hardware
 * performance metrics.
 */
class FieldDiagnostic final : public CoreStepActionInterface,
                              public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Input = inp::FieldDiagnostic;
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using SPConstVolume = std::shared_ptr<VolumeParams const>;
    using StoreT
        = StreamStore<FieldDiagnosticParamsData, FieldDiagnosticStateData>;
    template<MemSpace M>
    using ParamsRef = FieldDiagnosticParamsData<Ownership::const_reference, M>;
    template<MemSpace M>
    using StateRef = FieldDiagnosticStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with particle and volume data
    FieldDiagnostic(ActionId id,
                    Input const& input,
                    SPConstParticle particle,
                    SPConstVolume volume,
                    size_type num_streams);

    //!@{
    //! \name Metadata interface

    //! Label for the action and output
    std::string_view label() const final { return "field-diagnostic"; }
    // Description of the action
    std::string_view description() const final;
    //!@}

    //!@{
    //! \name Step action interface

    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    // Bin the propagation records with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Bin the propagation records with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

    //!@{
    //! \name Output interface

    //! Category of data to write
    Category category() const final { return Category::result; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    //!@{
    //! \name Data access for the instrumented propagation kernel

    //! Access the binning parameters
    template<MemSpace M>
    ParamsRef<M> const& params() const
    {
        return store_.params<M>();
    }

    //! Access (and lazily create) the state for a stream
    template<MemSpace M>
    StateRef<M>& state(StreamId stream, size_type size) const
    {
        return store_.state<M>(stream, size);
    }
    //!@}

  private:
    ActionId id_;
    SPConstParticle particle_;
    SPConstVolume volume_;
    mutable StoreT store_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
