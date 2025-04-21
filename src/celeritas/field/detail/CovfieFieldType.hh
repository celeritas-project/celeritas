//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/CovfieFieldType.hh
//---------------------------------------------------------------------------//
#pragma once

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

#include "corecel/Config.hh"

#include "corecel/Types.hh"

#if CELERITAS_USE_CUDA
#    include <covfie/cuda/backend/primitive/cuda_texture.hpp>
#elif CELERITAS_USE_HIP
#    include <covfie/hip/backend/primitive/hip_device_array.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Covfie field type
template<MemSpace M>
struct CovfieFieldTrait;

template<>
struct CovfieFieldTrait<MemSpace::host>
{
    using storage_t = covfie::backend::array<covfie::vector::float3>;
    using storage_order_t
        = covfie::backend::strided<covfie::vector::size3, storage_t>;
    using interp_t = covfie::backend::linear<storage_order_t>;
    using coordinates_transform_t = covfie::backend::affine<interp_t>;
    using field_t = covfie::field<coordinates_transform_t>;
    using builder_t = covfie::field<storage_order_t>;
};

template<>
struct CovfieFieldTrait<MemSpace::device>
{
#if CELERITAS_USE_CUDA

    using storage_t = covfie::backend::cuda_texture<covfie::vector::float3,
                                                    covfie::vector::float3>;
    using coordinates_transform_t = covfie::backend::affine<storage_t>;
    using field_t = covfie::field<coordinates_transform_t>;

#elif CELERITAS_USE_HIP

    using storage_t = covfie::backend::hip_device_array<covfie::vector::float3,
                                                        covfie::vector::float3>;
    using storage_order_t
        = covfie::backend::strided<covfie::vector::size3, storage_t>;
    using interp_t = covfie::backend::linear<storage_order_t>;
    using coordinates_transform_t = covfie::backend::affine<interp_t>;
    using field_t = covfie::field<coordinates_transform_t>;

#else
    using field_t = CovfieFieldTrait<MemSpace::host>::field_t;
#endif
};

//---------------------------------------------------------------------------//
}  // namespace celeritas