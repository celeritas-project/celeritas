//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OpticalGeneratorBase.cc
//---------------------------------------------------------------------------//
#include "OpticalGeneratorBase.hh"

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with IDs, label, and description.
 */
OpticalGeneratorBase::OpticalGeneratorBase(
    ActionId id,
    AuxId aux_id,
    GeneratorId gen_id,
    std::string_view label,
    std::string_view description) noexcept(!CELERITAS_DEBUG)
    : sad_{id, label, description}, aux_id_(aux_id), gen_id_(gen_id)
{
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(gen_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Get generator counters (mutable).
 */
GeneratorStateBase& OpticalGeneratorBase::counters(AuxStateVec& aux) const
{
    return dynamic_cast<GeneratorStateBase&>(aux.at(aux_id_));
}

//---------------------------------------------------------------------------//
/*!
 * Get generator counters.
 */
GeneratorStateBase const&
OpticalGeneratorBase::counters(AuxStateVec const& aux) const
{
    return dynamic_cast<GeneratorStateBase const&>(aux.at(aux_id_));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
