//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/ActionRegistryImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionInterface;

namespace detail
{
//---------------------------------------------------------------------------//
//! Traits class for differentiating const from mutable actions.
template<class T, class = void>
struct ActionSpTraits
{
    static constexpr bool is_const_action = false;
    static constexpr bool is_mutable_action = false;
};
template<class T>
struct ActionSpTraits<T, std::enable_if_t<std::is_base_of_v<ActionInterface, T>>>
{
    static constexpr bool is_const_action = std::is_const_v<T>;
    static constexpr bool is_mutable_action = !is_const_action;
};

//---------------------------------------------------------------------------//
//! True if T is a const class inheriting from ActionInterface.
template<class T>
inline constexpr bool is_const_action_v = ActionSpTraits<T>::is_const_action;

//---------------------------------------------------------------------------//
//! True if T is a mutable class inheriting from ActionInterface.
template<class T>
inline constexpr bool is_mutable_action_v
    = ActionSpTraits<T>::is_mutable_action;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
