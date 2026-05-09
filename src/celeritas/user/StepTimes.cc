//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepTimes.cc
//---------------------------------------------------------------------------//
#include "StepTimes.hh"

#include "corecel/data/AuxParamsRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to the aux registry.
 */
std::shared_ptr<StepTimes>
StepTimes::make_and_insert(SPAuxParamsRegistry const& aux, std::string label)
{
    auto result = std::make_shared<StepTimes>(aux->next_id(), std::move(label));
    aux->insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from ID and label.
 */
StepTimes::StepTimes(AuxId aux_id, std::string label)
    : aux_id_(aux_id), label_(std::move(label))
{
    CELER_EXPECT(aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Build core state data for a stream.
 */
auto StepTimes::create_state(MemSpace, StreamId, size_type) const -> UPState
{
    //! \todo Construct with limits.step_iters to limit capacity?
    constexpr size_type min_alloc{65536};
    auto result = std::make_unique<StepTimesState>();
    result->time.reserve(min_alloc);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access the state.
 */
StepTimesState const& StepTimes::state(AuxStateVec const& aux) const
{
    return get<StepTimesState>(aux, aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Access the state (mutable).
 */
StepTimesState& StepTimes::state(AuxStateVec& aux) const
{
    return get<StepTimesState>(aux, aux_id_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
