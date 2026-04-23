//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/CovfieRZFieldTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/clamp.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

#if CELERITAS_USE_CUDA
#    include <covfie/cuda/backend/primitive/cuda_device_array.hpp>
#elif CELERITAS_USE_HIP
#    include <covfie/hip/backend/primitive/hip_device_array.hpp>
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Covfie 2D field type for RZ maps
template<MemSpace M>
struct CovfieRZFieldTraits;

template<>
struct CovfieRZFieldTraits<MemSpace::host>
{
    using storage_t = covfie::backend::array<covfie::vector::float2>;
    using dimensioned_t
        = covfie::backend::strided<covfie::vector::size2, storage_t>;
    using interp_t = covfie::backend::linear<dimensioned_t>;
    using clamped_t = covfie::backend::clamp<interp_t>;
    using transformed_t = covfie::backend::affine<clamped_t>;
    using field_t = covfie::field<transformed_t>;
    using builder_t = covfie::field<dimensioned_t>;

    static Array<real_type, 2> to_array(field_t::output_t const& vec)
    {
        return {vec[0], vec[1]};
    }
};

template<>
struct CovfieRZFieldTraits<MemSpace::device>
{
    using float2 = covfie::vector::float2;
#if CELERITAS_USE_CUDA
    using storage_t = covfie::backend::cuda_device_array<float2>;
#elif CELERITAS_USE_HIP
    using storage_t = covfie::backend::hip_device_array<float2>;
#else
    using storage_t = covfie::backend::array<float2>;
#endif
    using dimensioned_t
        = covfie::backend::strided<covfie::vector::size2, storage_t>;
    using interp_t = covfie::backend::linear<dimensioned_t>;
    using clamped_t = covfie::backend::clamp<interp_t>;
    using transformed_t = covfie::backend::affine<clamped_t>;
    using field_t = covfie::field<transformed_t>;

    CELER_FUNCTION static Array<real_type, 2>
    to_array(field_t::output_t const& vec)
    {
        return {vec[0], vec[1]};
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
