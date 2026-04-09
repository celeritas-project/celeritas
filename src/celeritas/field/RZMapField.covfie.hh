//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapField.covfie.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Types.hh"

#include "RZMapFieldData.covfie.hh"  // IWYU pragma: keep

#include "detail/CovfieRZFieldTraits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Return the covfie view for the given params.
 *
 * This must be a function template (M as a template parameter) so that
 * \c if constexpr actually suppresses instantiation of the discarded branch.
 * In the device specialization, \c params.field_view is a \c void const*
 * that must be cast to the concrete view type; in the host specialization,
 * \c params.get_view() returns a stored \c view_t const& directly.
 */
template<MemSpace M>
CELER_FUNCTION auto
rzmap_get_view(RZMapFieldParamsData<Ownership::const_reference, M> const& params)
    -> typename CovfieRZFieldTraits<M>::field_t::view_t const&
{
    if constexpr (M == MemSpace::device)
    {
        using view_t = typename CovfieRZFieldTraits<M>::field_t::view_t;
        return *static_cast<view_t const*>(params.field_view);
    }
    else
    {
        return params.get_view();
    }
}
//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a volume-based RZ field map.
 *
 * \warning Accessing values outside the grid clamps to boundary values.
 * This behavior differs from the non-covfie implementation, where values
 * outside the map are assumed zero.
 */
class RZMapField
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = float;
    using Real3 = Array<celeritas::real_type, 3>;
    using ParamsRef = NativeCRef<RZMapFieldParamsData>;
    //!@}

  public:
    // Construct with the shared map data
    inline CELER_FUNCTION explicit RZMapField(ParamsRef const& shared);

    // Evaluate the magnetic field value for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 const& pos) const;

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data.
 */
CELER_FUNCTION
RZMapField::RZMapField(ParamsRef const& shared) : params_{shared} {}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector for the given position.
 *
 * This queries the covfie 2D field at (r, z) to obtain (Br, Bz), then
 * projects Br onto the x and y components using the position direction.
 * The result is in the native Celeritas unit system.
 */
CELER_FUNCTION auto RZMapField::operator()(Real3 const& pos) const -> Real3
{
    using traits_t = detail::CovfieRZFieldTraits<MemSpace::native>;
    celeritas::real_type r = hypot(pos[0], pos[1]);

    auto const& view = detail::rzmap_get_view<MemSpace::native>(params_);
    auto bvec = traits_t::to_array(
        view.at(static_cast<real_type>(r), static_cast<real_type>(pos[2])));

    // bvec = {Br, Bz}
    Real3 value;
    value[2] = bvec[1];

    celeritas::real_type br_over_r = (r != 0) ? bvec[0] / r
                                              : celeritas::real_type(0);
    value[0] = br_over_r * pos[0];
    value[1] = br_over_r * pos[1];

    return value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
