//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/detail/BuiltinSurfaceModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <string_view>
#include <vector>

#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Utility for building built-in surface models from input data.
 *
 * Wraps the call to build a model with a check on whether the input data is
 * empty. If empty, then the model is not built. Keeps track of number of
 * models built and constructs new models with the next ID.
 */
class BuiltinSurfaceModelBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<SurfaceModel>;
    //!@}

  public:
    // Construct with storage to fill
    explicit inline BuiltinSurfaceModelBuilder(std::vector<SPModel>& models);

    // Construct a built-in surface model
    template<class M>
    inline void build(std::map<PhysSurfaceId, typename M::InputT> const&);

    // Number of physics surfaces that have been constructed
    size_type num_surfaces() const { return num_surf_; }

  private:
    std::vector<SPModel>& models_;
    size_type num_surf_{0};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
BuiltinSurfaceModelBuilder::BuiltinSurfaceModelBuilder(
    std::vector<SPModel>& model)
    : models_(model)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a built-in surface model.
 *
 * Only constructs and adds the surface model if the \c layer_map is not empty.
 */
template<class M>
void BuiltinSurfaceModelBuilder::build(
    std::map<PhysSurfaceId, typename M::InputT> const& layer_map)
{
    if (!layer_map.empty())
    {
        models_.push_back(
            std::make_shared<M>(SurfaceModelId(models_.size()), layer_map));
        num_surf_ += layer_map.size();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
