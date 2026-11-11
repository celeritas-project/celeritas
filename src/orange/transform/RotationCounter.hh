//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//----------------------------------G-----------------------------------------//
//! \file orange/transform/RotationCounter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <unordered_map>

#include "corecel/Types.hh"
#include "geocel/Types.hh"

#include "VariantTransform.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find the most common rotation among set of transformations.
 *
 * The most common rotation is used for universe alignment, such that an LV
 * is oriented with the plurality of it's daughters axis-aligned. Rotations are
 * hashed based on their exact values; there is no notion of soft equivalence.
 *
 * \todo Merge rotations that are soft equal (see similar todo in
 * \c TransformInserter ).
 */
class RotationCounter
{
  public:
    //// TYPES ////

    struct Results
    {
        SquareMatrixReal3 rotation;  //!< Most common rotation
        size_type count;  //!< Number of transforms with \c rotation
    };

  public:
    //// METHODS ////

    // Count a single transform's rotation
    void count(VariantTransform const& transform);

    // Calculate the most common rotation and the number of transforms sharing
    // it
    Results calc_most_common() const;

    //// ACCESSORS ////

    //! Get the number of distinct rotations
    size_type num_rotations() const { return counts_.size(); }

  private:
    //// TYPES ////

    struct RotationMatrixHasher
    {
        // Hash a rotation matrix using its 9 values
        std::size_t operator()(SquareMatrixReal3 const&) const;
    };

    //// DATA ////

    std::unordered_map<SquareMatrixReal3, size_type, RotationMatrixHasher>
        counts_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
