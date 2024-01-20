//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/SignedPermutation.cc
//---------------------------------------------------------------------------//
#include "SignedPermutation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an identity permutation.
 */
SignedPermutation::SignedPermutation()
    : SignedPermutation{
        SignedAxes{{{psign, Axis::x}, {psign, Axis::y}, {psign, Axis::z}}}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a permutation vector.
 */
SignedPermutation::SignedPermutation(SignedAxes permutation) : compressed_{0}
{
    EnumArray<Axis, bool> encountered_ax{false, false, false};

    for (auto ax : {Axis::z, Axis::y, Axis::x})
    {
        auto new_ax_sign = permutation[ax];
        CELER_VALIDATE(
            new_ax_sign.first == psign || new_ax_sign.first == msign,
            << "invalid permutation sign '" << new_ax_sign.first << "'");
        CELER_VALIDATE(new_ax_sign.second < Axis::size_,
                       << "invalid permutation axis");
        CELER_VALIDATE(!encountered_ax[new_ax_sign.second],
                       << "duplicate axis " << to_char(new_ax_sign.second)
                       << " in permutation matrix input");
        encountered_ax[new_ax_sign.second] = true;

        // Push back "flip bit"
        compressed_ <<= 1;
        compressed_ |= (new_ax_sign.first == msign ? 0b1 : 0b0);
        // Push back "new axis"
        compressed_ <<= 2;
        compressed_ |= to_int(new_ax_sign.second);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Reconstruct the permutation.
 */
auto SignedPermutation::permutation() const -> SignedAxes
{
    SignedAxes result;

    UIntT temp = compressed_;
    for (auto ax : range(Axis::size_))
    {
        // Copy "new axis"
        result[ax].second = to_axis(temp & 0b11);
        temp >>= 2;
        // Push back "flip bit"
        result[ax].first = temp & 0b1 ? msign : psign;
        temp >>= 1;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the data for type-deleted storage.
 */
auto SignedPermutation::data() const -> DataArray
{
    static_assert(
        max_value() == static_cast<UIntT>(static_cast<real_type>(max_value())),
        "cannot round-trip real type and short int");

    return {static_cast<real_type>(compressed_)};
}

//---------------------------------------------------------------------------//
/*!
 * Make a permutation by rotating about the given axis.
 */
SignedPermutation make_permutation(Axis ax, QuarterTurn theta)
{
    CELER_EXPECT(ax < Axis::size_);

    auto to_sign = [](int v) {
        if (v < 0)
            return SignedPermutation::msign;
        return SignedPermutation::psign;
    };

    int const cost = cos(theta);
    int const sint = sin(theta);
    Axis const uax = to_axis((to_int(ax) + 1) % 3);
    Axis const vax = to_axis((to_int(ax) + 2) % 3);

    SignedPermutation::SignedAxes r;

    // Axis of rotation is unchanged
    r[ax] = {to_sign(1), ax};
    if (cost != 0)
    {
        r[uax] = {to_sign(cost), uax};
        r[vax] = {to_sign(cost), vax};
    }
    else
    {
        r[uax] = {to_sign(-sint), vax};
        r[vax] = {to_sign(sint), uax};
    }

    return SignedPermutation{r};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas