//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnosticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/FieldPropagationCounters.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
/*!
 * Quantities summed over propagation steps in each energy bin and volume.
 *
 * The first four count steps by outcome (all steps, and those ending on a
 * boundary, looping, or stuck); the rest are sums of \c
 * FieldPropagationCounters .
 */
enum class FieldTally : size_type
{
    steps,
    boundary,
    looping,
    stuck,
    substeps,
    accepted,
    bumped,
    intercepted,
    shortened,
    integrations,
    first_substep_integrations,
    quick_advances,
    chord_iterations,
    accurate_advances,
    accurate_iterations,
    good_step_rejects,
    intersects,
    intersect_hits,
    skippable_intersects,
    safeties,
    size_
};

//---------------------------------------------------------------------------//
//! Quantities binned into a histogram for each propagation step
enum class FieldStepHist : size_type
{
    substeps,
    integrations,
    first_substep_integrations,
    intersects,
    size_
};

//---------------------------------------------------------------------------//
//! Quantities binned into a histogram for each track at the end of its life
enum class FieldTrackHist : size_type
{
    steps,
    substeps,
    integrations,
    intersects,
    size_
};

//---------------------------------------------------------------------------//
//! Quantities accumulated for each step iteration
enum class FieldStepIter : size_type
{
    tracks,
    substeps,
    integrations,
    intersects,
    max_substeps,
    max_integrations,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Result of a single field propagation, written by the propagate kernel.
 *
 * Each track slot owns one record, so the propagation kernel writes it without
 * atomics. The diagnostic action bins and invalidates it later in the step.
 */
struct FieldStepRecord
{
    FieldPropagationCounters counters;
    real_type step{0};  //!< Requested step length
    real_type distance{0};  //!< Distance actually propagated
    real_type start_safety{-1};  //!< Safety at the start, or -1 if unused
    real_type energy{0};  //!< Kinetic energy [MeV]
    VolumeId volume;  //!< Volume at the start of propagation
    bool boundary{false};  //!< Propagation ended on a boundary
    bool looping{false};  //!< Propagation reached the substep limit
    bool valid{false};  //!< Record has not yet been binned
};

//---------------------------------------------------------------------------//
//! Totals accumulated over the lifetime of a track
struct FieldTrackTally
{
    size_type steps{0};
    size_type substeps{0};
    size_type integrations{0};
    size_type intersects{0};
};

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Binning parameters for the field propagation diagnostic.
 *
 * Counts per step or per track are binned from 0 to \c num_bins - 2 with an
 * overflow bin. Energies and safety ratios are binned logarithmically with
 * an underflow bin (which includes zero) and an overflow bin.
 */
template<Ownership W, MemSpace M>
struct FieldDiagnosticParamsData
{
    //// TYPES ////

    using SizeId = ItemId<ull_int>;

    //// DATA ////

    //! Number of count bins, including overflow
    size_type num_bins{0};
    //! Number of particle types
    size_type num_particles{0};
    //! Number of canonical volumes
    size_type num_volumes{0};
    //! Maximum number of step iterations to record
    size_type num_step_records{0};
    //! Calculate the safety at the start of each step
    bool oracle_safety{false};

    //!@{
    //! Logarithmic energy binning [MeV]
    real_type log_energy_min{-3};
    real_type energy_bins_per_decade{5};
    size_type num_energy_bins{52};
    //!@}

    //!@{
    //! Logarithmic binning of the safety-to-step ratio
    real_type log_safety_min{-4};
    real_type safety_bins_per_decade{10};
    size_type num_safety_bins{62};
    //!@}

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_bins > 1 && num_particles > 0 && num_volumes > 0
               && num_energy_bins > 2 && num_safety_bins > 2;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    FieldDiagnosticParamsData& operator=(
        FieldDiagnosticParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        num_bins = other.num_bins;
        num_particles = other.num_particles;
        num_volumes = other.num_volumes;
        num_step_records = other.num_step_records;
        oracle_safety = other.oracle_safety;
        log_energy_min = other.log_energy_min;
        energy_bins_per_decade = other.energy_bins_per_decade;
        num_energy_bins = other.num_energy_bins;
        log_safety_min = other.log_safety_min;
        safety_bins_per_decade = other.safety_bins_per_decade;
        num_safety_bins = other.num_safety_bins;
        return *this;
    }

    //// INDEXING ////

    //! Integer value of a diagnostic enum
    template<class E>
    static CELER_CONSTEXPR_FUNCTION size_type index_of(E e)
    {
        return static_cast<size_type>(e);
    }

    //! Size of the per-step histograms
    CELER_FUNCTION size_type step_hist_size() const
    {
        return num_particles * index_of(FieldStepHist::size_) * num_bins;
    }

    //! Size of the per-track histograms
    CELER_FUNCTION size_type track_hist_size() const
    {
        return num_particles * index_of(FieldTrackHist::size_) * num_bins;
    }

    //! Size of the energy tally
    CELER_FUNCTION size_type energy_size() const
    {
        return num_energy_bins * index_of(FieldTally::size_);
    }

    //! Size of the volume tally
    CELER_FUNCTION size_type volume_size() const
    {
        return num_volumes * index_of(FieldTally::size_);
    }

    //! Size of the safety histogram
    CELER_FUNCTION size_type safety_size() const
    {
        return num_particles * num_safety_bins;
    }

    //! Size of the step iteration records
    CELER_FUNCTION size_type step_iter_size() const
    {
        return num_step_records * index_of(FieldStepIter::size_);
    }

    //! Index into a per-step histogram, clamping the count to the overflow
    CELER_FUNCTION SizeId step_hist(
        ParticleId pid, FieldStepHist q, size_type count) const
    {
        return SizeId{(pid.get() * index_of(FieldStepHist::size_) + index_of(q))
                          * num_bins
                      + celeritas::min(count, num_bins - 1)};
    }

    //! Index into a per-track histogram, clamping the count to the overflow
    CELER_FUNCTION SizeId track_hist(
        ParticleId pid, FieldTrackHist q, size_type count) const
    {
        return SizeId{
            (pid.get() * index_of(FieldTrackHist::size_) + index_of(q))
                * num_bins
            + celeritas::min(count, num_bins - 1)};
    }

    //! Index of the first tally in an energy bin
    CELER_FUNCTION SizeId energy_tally(real_type energy) const
    {
        size_type bin = log_bin(
            energy, log_energy_min, energy_bins_per_decade, num_energy_bins);
        return SizeId{bin * index_of(FieldTally::size_)};
    }

    //! Index of the first tally in a volume
    CELER_FUNCTION SizeId volume_tally(VolumeId vol) const
    {
        return SizeId{vol.get() * index_of(FieldTally::size_)};
    }

    //! Index into the safety histogram
    CELER_FUNCTION SizeId safety_hist(ParticleId pid, real_type ratio) const
    {
        return SizeId{pid.get() * num_safety_bins
                      + log_bin(ratio,
                                log_safety_min,
                                safety_bins_per_decade,
                                num_safety_bins)};
    }

    //! Index into the step iteration records
    CELER_FUNCTION SizeId step_iter(size_type iter, FieldStepIter q) const
    {
        return SizeId{iter * index_of(FieldStepIter::size_) + index_of(q)};
    }

    //! Logarithmic bin with underflow (including nonpositive) and overflow
    static CELER_FUNCTION size_type log_bin(real_type value,
                                            real_type log_min,
                                            real_type bins_per_decade,
                                            size_type num_bins)
    {
        if (!(value > 0))
        {
            return 0;
        }
        real_type x = (std::log10(value) - log_min) * bins_per_decade;
        if (!(x >= 0))
        {
            return 0;
        }
        if (!(x < static_cast<real_type>(num_bins - 2)))
        {
            return num_bins - 1;
        }
        return 1 + static_cast<size_type>(x);
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Per-slot records and accumulated tallies for a stream.
 *
 * The \c step_iter counter is incremented on the host after every step.
 */
template<Ownership W, MemSpace M>
struct FieldDiagnosticStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    //!@{
    //! \name Per track slot
    StateItems<FieldStepRecord> step;
    StateItems<FieldTrackTally> track;
    //!@}

    //!@{
    //! \name Accumulated over the run
    Items<ull_int> step_hist;
    Items<ull_int> track_hist;
    Items<ull_int> energy;
    Items<ull_int> volume;
    Items<ull_int> safety;
    Items<ull_int> step_iters;
    //!@}

    //! Index of the current step iteration
    size_type step_iter{0};

    //// METHODS ////

    //! Number of track slots
    CELER_FUNCTION size_type size() const { return step.size(); }

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !step.empty() && track.size() == step.size()
               && !step_hist.empty() && !track_hist.empty() && !energy.empty()
               && !volume.empty() && !safety.empty();
    }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    FieldDiagnosticStateData& operator=(FieldDiagnosticStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        step = other.step;
        track = other.track;
        step_hist = other.step_hist;
        track_hist = other.track_hist;
        energy = other.energy;
        volume = other.volume;
        safety = other.safety;
        step_iters = other.step_iters;
        step_iter = other.step_iter;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Resize and zero the state
template<MemSpace M>
void resize(FieldDiagnosticStateData<Ownership::value, M>* state,
            HostCRef<FieldDiagnosticParamsData> const& params,
            StreamId,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace celeritas
