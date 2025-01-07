//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
ActionInterface::~ActionInterface() noexcept = default;

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from a label and ID.
 */
ConcreteAction::ConcreteAction(ActionId id,
                               std::string label) noexcept(!CELERITAS_DEBUG)
    : ConcreteAction{id, std::move(label), {}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from an ID, a unique label, and a description.
 */
ConcreteAction::ConcreteAction(ActionId id,
                               std::string label,
                               std::string description) noexcept(!CELERITAS_DEBUG)
    : id_{std::move(id)}
    , label_{std::move(label)}
    , description_{std::move(description)}
{
    CELER_ASSERT(id_);
    CELER_ASSERT(!label_.empty());
}

//---------------------------------------------------------------------------//
//! Default destructor.
ConcreteAction::~ConcreteAction() noexcept = default;

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from a label and ID.
 */
StaticActionData::StaticActionData(
    ActionId id, std::string_view label) noexcept(!CELERITAS_DEBUG)
    : StaticActionData{id, label, {}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from an ID, a unique label, and a description.
 */
StaticActionData::StaticActionData(
    ActionId id,
    std::string_view label,
    std::string_view description) noexcept(!CELERITAS_DEBUG)
    : id_{id}, label_{label}, description_{description}
{
    CELER_ASSERT(id_);
    CELER_ASSERT(!label_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from a label and ID.
 */
StaticConcreteAction::StaticConcreteAction(
    ActionId id, std::string_view label) noexcept(!CELERITAS_DEBUG)
    : sad_{id, label}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a concrete action from an ID, a unique label, and a description.
 */
StaticConcreteAction::StaticConcreteAction(
    ActionId id,
    std::string_view label,
    std::string_view description) noexcept(!CELERITAS_DEBUG)
    : sad_{id, label, description}
{
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an action order.
 */
char const* to_cstring(StepActionOrder value)
{
    static EnumStringMapper<StepActionOrder> const to_cstring_impl{
        "generate",
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
