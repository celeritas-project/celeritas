//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/VariantSurfacesImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "../SurfaceTypeTraits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

//! Underlying integer for surface type enum
using STInt = std::underlying_type_t<SurfaceType>;

//! Compile-time range of all surfaces
using SurfaceTypeIntRange
    = std::make_integer_sequence<STInt, static_cast<STInt>(SurfaceType::size_)>;

//! Get surface type from the enum value rather than the enum
template<STInt I>
struct SurfaceTypeIntTraits
{
    using type = typename SurfaceTypeTraits<static_cast<SurfaceType>(I)>::type;
};

//!@{
//! Helper for expanding an integer sequence into a variant
template<class Seq>
struct VariantSurfacesImpl;
template<STInt... Is>
struct VariantSurfacesImpl<std::integer_sequence<STInt, Is...>>
{
    using type = std::variant<typename SurfaceTypeIntTraits<Is>::type...>;
};
//!@}

//! Get a variant with all the surface types
using VariantSurfaces_t =
    typename VariantSurfacesImpl<SurfaceTypeIntRange>::type;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
