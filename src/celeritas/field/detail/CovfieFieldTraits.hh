//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/CovfieFieldTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "geocel/Types.hh"

#if CELERITAS_USE_CUDA
#    include <covfie/cuda/backend/primitive/cuda_texture.hpp>
#elif CELERITAS_USE_HIP
#    include <covfie/hip/backend/primitive/hip_device_array.hpp>
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Covfie field type
template<MemSpace M>
struct CovfieFieldTraits;

template<>
struct CovfieFieldTraits<MemSpace::host>
{
    using storage_t = covfie::backend::array<covfie::vector::float3>;
    using dimensioned_t
        = covfie::backend::strided<covfie::vector::size3, storage_t>;
    using interp_t = covfie::backend::linear<dimensioned_t>;
    using transformed_t = covfie::backend::affine<interp_t>;
    using field_t = covfie::field<transformed_t>;
    using builder_t = covfie::field<dimensioned_t>;

    static Real3 to_array(field_t::output_t const& vec)
    {
        return {vec[0], vec[1], vec[2]};
    }
};

template<>
struct CovfieFieldTraits<MemSpace::device>
{
    using float3 = covfie::vector::float3;
#if CELERITAS_USE_CUDA
    using storage_t = covfie::backend::cuda_texture<float3, float3>;
    using transformed_t = covfie::backend::affine<storage_t>;
#else
// Manually interpolate rather than relying on texture memory
#    if CELERITAS_USE_HIP
    using storage_t = covfie::backend::hip_device_array<float3, float3>;
#    else
    using storage_t = covfie::backend::array<float3>;
#    endif

    using dimensioned_t
        = covfie::backend::strided<covfie::vector::size3, storage_t>;
    using interp_t = covfie::backend::linear<dimensioned_t>;
    using transformed_t = covfie::backend::affine<interp_t>;
#endif

    using field_t = covfie::field<transformed_t>;

    CELER_FUNCTION static Real3 to_array(field_t::output_t const& vec)
    {
        return {vec[0], vec[1], vec[2]};
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
