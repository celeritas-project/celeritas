//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionInterface.cc
//---------------------------------------------------------------------------//
#include "ActionInterface.hh"

#include <utility>

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
}  // namespace celeritas
