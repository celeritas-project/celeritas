//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cu
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

#include "corecel/data/Collection.hh"

#include "StepData.hh"

using thrust::device_pointer_cast;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
using DVecDetector = thrust::device_vector<DetectorId>;

template<class T>
using StateRef
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
struct HasDetector
{
    CELER_FORCEINLINE_FUNCTION bool operator()(DetectorId const& d)
    {
        return static_cast<bool>(d);
    }
};

//---------------------------------------------------------------------------//
size_type count_num_valid(DVecDetector const& orig_ids)
{
    return thrust::count_if(
        thrust::device, orig_ids.begin(), orig_ids.end(), HasDetector{});
}

//---------------------------------------------------------------------------//
// Simplify opaque ID pointers to reduce code bloat from identical
// instantiations
template<class T>
struct PointerTransformer
{
    constexpr T* operator()(T* v) const { return v; }
};

template<class V, class S>
struct PointerTransformer<OpaqueId<V, S>>
{
    constexpr S* operator()(OpaqueId<V, S>* v) const
    {
        return reinterpret_cast<S*>(v);
    }
};

//---------------------------------------------------------------------------//
template<class T>
void assign_field(std::vector<T>* dst,
                  StateRef<T> const& src,
                  DVecDetector const& orig_ids,
                  size_type size)
{
    if (src.empty())
    {
        // This attribute is not in use
        dst->clear();
        return;
    }

    PointerTransformer<T> transform_ptr;

    // Partition based on detector validity
    auto temp_span = src[AllItems<T>{}];
    thrust::partition(thrust::device,
                      transform_ptr(temp_span.begin()),
                      transform_ptr(temp_span.end()),
                      orig_ids.begin(),
                      HasDetector{});

    // Copy all items from valid threads
    dst->resize(size);
    thrust::copy(device_pointer_cast(transform_ptr(temp_span.begin())),
                 device_pointer_cast(transform_ptr(temp_span.begin()) + size),
                 transform_ptr(dst->data()));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Copy to host results from tracks that interacted with a detector.
 *
 * \warning this mutates the original, but the mutated version will still have
 * consistent values for all tracks with valid detector IDs.
 */
template<>
void copy_steps<MemSpace::device>(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::device> const& state)
{
    CELER_EXPECT(output);

    // Copy the original detector IDs
    thrust::device_vector<DetectorId> orig_ids;
    {
        auto detector_span = state.detector[AllItems<DetectorId>{}];
        orig_ids.assign(device_pointer_cast(detector_span.begin()),
                        device_pointer_cast(detector_span.end()));
    }

    // Get the number of threads that are active and in a detector
    size_type size = count_num_valid(orig_ids);

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    assign_field(&(output->FIELD), state.FIELD, orig_ids, size)

    DS_ASSIGN(detector);
    DS_ASSIGN(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_ASSIGN(points[sp].time);
        DS_ASSIGN(points[sp].pos);
        DS_ASSIGN(points[sp].dir);
        DS_ASSIGN(points[sp].energy);
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(particle);
    DS_ASSIGN(energy_deposition);
#undef DS_ASSIGN

    CELER_ENSURE(output->detector.size() == size);
    CELER_ENSURE(output->track_id.size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
