//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Translator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayOperators.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Translate points from a parent's reference frame into the daughter.
 *
 * The input "translation" is the transformation applied to a daughter universe
 * to place it in the parent. The daughter is a "lower" level compared to the
 * parent.
 */
class TranslatorDown
{
  public:
    //! Construct with the parent-to-daughter translation
    explicit CELER_FUNCTION TranslatorDown(Real3 const& t) : trans_{t} {}

    //! Translate a single point
    CELER_FORCEINLINE_FUNCTION Real3 operator()(Real3 const& parent) const
    {
        return parent - trans_;
    }

  private:
    Real3 trans_;
};

//---------------------------------------------------------------------------//
/*!
 * Translate points from a daughter's reference frame "up" into the parent.
 *
 * The input "translation" is the transformation applied to a daughter universe
 * to place it in the parent. The daughter is a "lower" level compared to the
 * parent.
 */
class TranslatorUp
{
  public:
    //! Construct with the parent-to-daughter translation
    explicit CELER_FUNCTION TranslatorUp(Real3 const& t) : trans_{t} {}

    //! Translate a single point
    CELER_FORCEINLINE_FUNCTION Real3 operator()(Real3 const& parent) const
    {
        return parent + trans_;
    }

  private:
    Real3 trans_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
