//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Translator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Range.hh"
#include "orange/Types.hh"

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
    // Construct with the parent-to-daughter translation
    explicit inline CELER_FUNCTION TranslatorDown(const Real3& translation);

    // Translate a single point
    CELER_FORCEINLINE_FUNCTION Real3 operator()(const Real3& parent) const;

  private:
    const Real3& translation_;
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
    // Construct with the parent-to-daughter translation
    inline CELER_FUNCTION TranslatorUp(const Real3& translation);

    // Translate a single point
    CELER_FORCEINLINE_FUNCTION Real3 operator()(const Real3& parent) const;

  private:
    const Real3& translation_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with translation.
 */
CELER_FUNCTION TranslatorDown::TranslatorDown(const Real3& translation)
    : translation_(translation)
{
}

//---------------------------------------------------------------------------//
/*!
 * Translate a single point.
 */
CELER_FUNCTION Real3 TranslatorDown::operator()(const Real3& parent) const
{
    Real3 daughter;
    for (int i : range(3))
    {
        daughter[i] = parent[i] - translation_[i];
    }
    return daughter;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with translation.
 */
CELER_FUNCTION TranslatorUp::TranslatorUp(const Real3& translation)
    : translation_(translation)
{
}

//---------------------------------------------------------------------------//
/*!
 * Translate a single point.
 */
CELER_FUNCTION Real3 TranslatorUp::operator()(const Real3& parent) const
{
    Real3 daughter;
    for (int i : range(3))
    {
        daughter[i] = parent[i] + translation_[i];
    }
    return daughter;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
