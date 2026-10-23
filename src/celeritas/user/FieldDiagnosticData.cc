//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnosticData.cc
//---------------------------------------------------------------------------//
#include "FieldDiagnosticData.hh"

#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Copier.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Resize a collection and fill it with value-initialized elements.
 *
 * The device filler supports only a few types, so copy from host instead.
 */
template<class T, MemSpace M, class I>
void resize_init(Collection<T, Ownership::value, M, I>* col, size_type size)
{
    resize(col, size);
    std::vector<T> init(size);
    Copier<T, M> copy{(*col)[AllItems<T, M>{}]};
    copy(MemSpace::host, make_span(std::as_const(init)));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Resize and zero the per-slot records and accumulated tallies.
 */
template<MemSpace M>
void resize(FieldDiagnosticStateData<Ownership::value, M>* state,
            HostCRef<FieldDiagnosticParamsData> const& params,
            StreamId,
            size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);

    resize_init(&state->step, size);
    resize_init(&state->track, size);
    resize_init(&state->step_hist, params.step_hist_size());
    resize_init(&state->track_hist, params.track_hist_size());
    resize_init(&state->energy, params.energy_size());
    resize_init(&state->volume, params.volume_size());
    resize_init(&state->safety, params.safety_size());
    if (params.num_step_records > 0)
    {
        resize_init(&state->step_iters, params.step_iter_size());
    }
    state->step_iter = 0;

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template void resize(
    FieldDiagnosticStateData<Ownership::value, MemSpace::host>* state,
    HostCRef<FieldDiagnosticParamsData> const& params,
    StreamId,
    size_type);

template void resize(
    FieldDiagnosticStateData<Ownership::value, MemSpace::device>* state,
    HostCRef<FieldDiagnosticParamsData> const& params,
    StreamId,
    size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
