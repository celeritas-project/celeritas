//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/detail/BuiltinSurfaceModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/optical/surface/model/BuiltinSurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
template<>
struct BuiltinApplier<SurfacePhysicsOrder::size_>
{
    template<class T>
    using Applier = TrivialApplier<T>;
};

namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Fake model as a placeholder for surface models yet to be implemented.
 */
template<class T>
class FakeModel : public BuiltinSurfaceModel<SurfacePhysicsOrder::size_>
{
  public:
    FakeModel(SurfaceModelId model_id,
              std::string_view label,
              std::vector<PhysSurfaceId> surfaces)
        : BuiltinSurfaceModel<SurfacePhysicsOrder::size_>(
              model_id, label, std::move(surfaces))
    {
    }

    void step(CoreParams const&, CoreStateHost&) const final {}
    void step(CoreParams const&, CoreStateDevice&) const final {}
};

//---------------------------------------------------------------------------//
/*!
 * Build a fake model from input.
 */
template<class T>
std::shared_ptr<FakeModel<T>>
fake_builtin_model_from_input(SurfaceModelId model_id,
                              std::string_view label,
                              std::map<PhysSurfaceId, T> const& layer_map)
{
    std::vector<PhysSurfaceId> surfaces;
    for (auto const& [layer, input] : layer_map)
    {
        CELER_EXPECT(layer);
        CELER_DISCARD(input);
        surfaces.push_back(layer);
    }

    CELER_ENSURE(surfaces.size() == layer_map.size());

    return std::make_shared<FakeModel<T>>(model_id, label, std::move(surfaces));
}

}  // namespace

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

    // Construct a fake surface model
    template<class T>
    void build_fake(std::string_view label, std::map<PhysSurfaceId, T> const&);

    // Construct a built-in surface model
    template<class M>
    void build(std::map<PhysSurfaceId, typename M::InputT> const&);

    // Number of physics surfaces that have been constructed
    size_type num_surfaces() const { return num_surf_; }

  private:
    std::vector<SPModel>& models_;
    size_type num_surf_{0};

    // Construct built-in surface model from input
    template<class M>
    std::shared_ptr<M> builtin_model_from_input(
        SurfaceModelId, std::map<PhysSurfaceId, typename M::InputT> const&);
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
        models_.push_back(builtin_model_from_input<M>(
            SurfaceModelId(models_.size()), layer_map));
        num_surf_ += layer_map.size();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct a fake surface model.
 *
 * A temporary utility to build fake surface models that have not yet been
 * implemented.
 */
template<class T>
void BuiltinSurfaceModelBuilder::build_fake(
    std::string_view label, std::map<PhysSurfaceId, T> const& layer_map)
{
    if (!layer_map.empty())
    {
        models_.push_back(fake_builtin_model_from_input(
            SurfaceModelId(models_.size()), label, layer_map));
        num_surf_ += layer_map.size();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct built-in model of type \c M from \c inp::SurfacePhysics surface
 * layer map.
 *
 * The model \c M should have a type alias \c M::InputT which corresponds to
 * the input type associated with each surface.
 */
template<class M>
std::shared_ptr<M> BuiltinSurfaceModelBuilder::builtin_model_from_input(
    SurfaceModelId model_id,
    std::map<PhysSurfaceId, typename M::InputT> const& layer_map)
{
    std::vector<PhysSurfaceId> surfaces;
    std::vector<typename M::InputT> inputs;

    for (auto const& [surface, input] : layer_map)
    {
        CELER_ENSURE(surface);
        surfaces.push_back(surface);
        inputs.push_back(input);
    }

    CELER_ENSURE(surfaces.size() == layer_map.size());
    CELER_ENSURE(inputs.size() == layer_map.size());

    return std::make_shared<M>(model_id, std::move(surfaces), inputs);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
