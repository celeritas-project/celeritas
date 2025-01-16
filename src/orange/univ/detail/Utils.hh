//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "orange/OrangeTypes.hh"

#include "Types.hh"
#include "../VolumeView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// FUNCTION-LIKE CLASSES
//---------------------------------------------------------------------------//
/*!
 * Predicate for partitioning valid (finite positive) from invalid distances.
 */
struct IsFinite
{
    CELER_FORCEINLINE_FUNCTION bool operator()(real_type distance) const
    {
        return distance < numeric_limits<real_type>::max();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Predicate for selecting distances closer to or equal to a maximum.
 */
class IsNotFurtherThan
{
  public:
    explicit CELER_FORCEINLINE_FUNCTION IsNotFurtherThan(real_type md)
        : max_dist_(md)
    {
    }

    CELER_FORCEINLINE_FUNCTION bool operator()(real_type distance) const
    {
        return distance <= max_dist_;
    }

  private:
    real_type max_dist_;
};

//---------------------------------------------------------------------------//
/*!
 * Calculate the bump distance for a position.
 */
class BumpCalculator
{
  public:
    explicit CELER_FORCEINLINE_FUNCTION BumpCalculator(Tolerance<> const& tol)
        : tol_(tol)
    {
    }

    CELER_FUNCTION real_type operator()(Real3 const& pos) const
    {
        real_type result = tol_.abs;
        for (real_type p : pos)
        {
            result = celeritas::max(result, tol_.rel * std::fabs(p));
        }
        CELER_ENSURE(result > 0);
        return result;
    }

  private:
    Tolerance<> tol_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Convert an OnLocalSurface (may be null) to an OnFace using a volume view.
 */
inline CELER_FUNCTION OnFace find_face(VolumeView const& vol,
                                       OnLocalSurface surf)
{
    return {surf ? vol.find_face(surf.id()) : FaceId{}, surf.unchecked_sense()};
}

//---------------------------------------------------------------------------//
/*!
 * Convert an OnFace (may be null) to an OnLocalSurface using a volume view.
 */
inline CELER_FUNCTION OnLocalSurface get_surface(VolumeView const& vol,
                                                 OnFace face)
{
    return {face ? vol.get_surface(face.id()) : LocalSurfaceId{},
            face.unchecked_sense()};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a postfix logic expression to an infix expression.
 */
inline std::vector<logic_int> convert_to_infix(Span<logic_int> postfix)
{
    CELER_EXPECT(!postfix.empty());

    std::vector<std::vector<logic_int>> infix;
    infix.reserve(postfix.size());

    // Process each token
    for (auto lgc : postfix)
    {
        if (logic::is_operator_token(lgc))
        {
            switch (lgc)
            {
                case logic::ltrue:
                    infix.push_back({lgc});
                    break;
                case logic::lor:
                    [[fallthrough]];
                case logic::land: {
                    auto op_1 = infix.back();
                    auto op_2 = infix.back();
                    std::vector<logic_int> new_expr;
                    new_expr.reserve(3 + op_1.size() + op_2.size());

                    new_expr.push_back(logic::lopen);
                    new_expr.insert(new_expr.end(), op_1.begin(), op_1.end());
                    new_expr.push_back(lgc);
                    new_expr.insert(new_expr.end(), op_2.begin(), op_2.end());
                    new_expr.push_back(logic::lclose);

                    infix.pop_back();
                    infix.pop_back();
                    infix.push_back(new_expr);
                    break;
                }
                case logic::lnot: {
                    auto op = infix.back();
                    std::vector<logic_int> new_expr;
                    new_expr.reserve(1 + op.size());

                    new_expr.push_back(lgc);
                    new_expr.insert(new_expr.end(), op.begin(), op.end());

                    infix.pop_back();
                    infix.push_back(new_expr);
                    break;
                }
                default:
                    CELER_ASSERT_UNREACHABLE();
            }
        }
        else
        {
            infix.push_back({lgc});
        }
    }
    CELER_ENSURE(infix.size() == 1);
    return infix.front();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
