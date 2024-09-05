//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ActionInterface.cc
//---------------------------------------------------------------------------//
#include "ActionInterface.hh"

#include <utility>

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Default destructor.
ActionInterface::~ActionInterface() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from a label and ID.
 */
ConcreteAction::ConcreteAction(ActionId id, std::string label)
    : ConcreteAction{id, std::move(label), {}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from an ID, a unique label, and a description.
 */
ConcreteAction::ConcreteAction(ActionId id,
                               std::string label,
                               std::string description)
    : id_{std::move(id)}
    , label_{std::move(label)}
    , description_{std::move(description)}
{
    CELER_ASSERT(id_);
    CELER_ASSERT(!label_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an action order.
 */
char const* to_cstring(StepActionOrder value)
{
    static EnumStringMapper<StepActionOrder> const to_cstring_impl{
        "start",
        "user_start",
        "sort_start",
        "pre",
        "user_pre",
        "sort_pre",
        "along",
        "sort_along",
        "pre_post",
        "sort_pre_post",
        "post",
        "user_post",
        "end",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
