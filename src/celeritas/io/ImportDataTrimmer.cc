//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportDataTrimmer.cc
//---------------------------------------------------------------------------//
#include "ImportDataTrimmer.hh"

#include <algorithm>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct ImportDataTrimmer::GridFilterer
{
    size_type stride;
    size_type orig_size;

    // Whether to keep the data at this index
    inline bool operator()(size_type i) const;
};

//---------------------------------------------------------------------------//
/*!
 * Construct from input parameters.
 */
ImportDataTrimmer::ImportDataTrimmer(Input const& inp) : options_{inp}
{
    CELER_EXPECT(options_.max_size >= 2);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportData* data)
{
    if (options_.materials)
    {
        // Reduce the number of materials, elements, etc.
        (*this)(&data->isotopes);
        (*this)(&data->elements);
        (*this)(&data->geo_materials);
        (*this)(&data->phys_materials);

        (*this)(&data->regions);
        (*this)(&data->volumes);

        (*this)(&data->sb_data);
        (*this)(&data->livermore_pe_data);
        (*this)(&data->neutron_elastic_data);
        (*this)(&data->atomic_relaxation_data);

        (*this)(&data->optical_materials);
    }

    if (options_.physics)
    {
        // Reduce the number of physics processes
        (*this)(&data->particles);
        (*this)(&data->processes);
        (*this)(&data->msc_models);
    }

    if (options_.mupp)
    {
        // Reduce the resolution of the muon pair production table
        (*this)(&data->mu_pair_production_data);
    }

    this->for_each(&data->elements);
    this->for_each(&data->geo_materials);
    this->for_each(&data->phys_materials);

    this->for_each(&data->processes);
    this->for_each(&data->msc_models);
    this->for_each(&data->sb_data);
    this->for_each(&data->livermore_pe_data);
    this->for_each(&data->neutron_elastic_data);
    this->for_each(&data->atomic_relaxation_data);

    this->for_each(&data->optical_models);
    this->for_each(&data->optical_materials);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportElement* data)
{
    CELER_EXPECT(data);

    if (options_.materials)
    {
        (*this)(&data->isotopes_fractions);
    }
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportGeoMaterial* data)
{
    CELER_EXPECT(data);

    if (options_.materials)
    {
        (*this)(&data->elements);
    }
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportPhysMaterial* data)
{
    CELER_EXPECT(data);

    // TODO: remap IDs?
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportOpticalMaterial* data)
{
    CELER_EXPECT(data);

    // TODO: trim WLS components?
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportOpticalModel* data)
{
    CELER_EXPECT(data);

    if (options_.materials)
    {
        (*this)(&data->mfp_table);
    }

    this->for_each(data->mfp_table);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportModelMaterial* data)
{
    CELER_EXPECT(data);

    (*this)(&data->energy);

    if (options_.materials)
    {
        (*this)(&data->micro_xs);
    }

    this->for_each(data->micro_xs);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportModel* data)
{
    CELER_EXPECT(data);

    if (options_.materials)
    {
        (*this)(&data->materials);
    }

    this->for_each(data->materials);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportMscModel* data)
{
    CELER_EXPECT(data);

    (*this)(&data->xs_table);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportMuPairProductionTable* data)
{
    CELER_EXPECT(data);

    if (!*data)
    {
        return;
    }

    (*this)(&data->atomic_number);
    (*this)(&data->physics_vectors);

    this->for_each(&data->physics_vectors);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportParticle* data)
{
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportProcess* data)
{
    CELER_EXPECT(data);

    if (options_.materials)
    {
        (*this)(&data->tables);
    }
    if (options_.physics)
    {
        (*this)(&data->models);
    }

    this->for_each(&data->models);
    this->for_each(&data->tables);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportPhysicsVector* data)
{
    CELER_EXPECT(data);
    (*this)(&data->x);
    (*this)(&data->y);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportPhysicsTable* data)
{
    CELER_EXPECT(data);
    if (options_.materials)
    {
        (*this)(&data->physics_vectors);
    }

    this->for_each(&data->physics_vectors);
}

//---------------------------------------------------------------------------//
void ImportDataTrimmer::operator()(ImportPhysics2DVector* data)
{
    auto x_filter = this->make_filterer(data->x.size());
    auto y_filter = this->make_filterer(data->y.size());

    // Trim x and y grid
    (*this)(&data->x);
    (*this)(&data->y);

    std::vector<double> new_value;
    new_value.reserve(data->x.size() * data->y.size());

    auto src = data->value.cbegin();
    for (auto i : range(x_filter.orig_size))
    {
        for (auto j : range(y_filter.orig_size))
        {
            if (x_filter(i) && y_filter(j))
            {
                new_value.push_back(*src);
            }
            ++src;
        }
    }
    CELER_ASSERT(src == data->value.cend());

    data->value = std::move(new_value);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
/*!
 * Trim the number of elements in a vector.
 */
template<class T>
void ImportDataTrimmer::operator()(std::vector<T>* vec)
{
    CELER_EXPECT(vec);

    if (options_.max_size == numeric_limits<size_type>::max())
    {
        // Don't trim
        return;
    }

    std::vector<T> result;
    result.reserve(std::min(options_.max_size + 1, vec->size()));

    auto filter = this->make_filterer(vec->size());
    for (auto i : range(vec->size()))
    {
        if (filter(i))
        {
            result.push_back((*vec)[i]);
        }
    }
    *vec = std::move(result);
}

//---------------------------------------------------------------------------//
/*!
 * Trim the number of elements in a map.
 */
template<class K, class T, class C, class A>
void ImportDataTrimmer::operator()(std::map<K, T, C, A>* m)
{
    CELER_EXPECT(m);

    std::map<K, T, C, A> result;
    auto filter = this->make_filterer(m->size());
    auto iter = m->begin();
    for (auto i : range(filter.orig_size))
    {
        auto cur_iter = iter++;
        if (filter(i))
        {
            result.insert(m->extract(cur_iter));
        }
    }
    *m = std::move(result);
}

//---------------------------------------------------------------------------//
/*!
 * Trim each element in a vector.
 */
template<class T>
void ImportDataTrimmer::for_each(std::vector<T>* vec)
{
    CELER_EXPECT(vec);

    for (auto& v : *vec)
    {
        (*this)(&v);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Trim each value in a map.
 */
template<class K, class T, class C, class A>
void ImportDataTrimmer::for_each(std::map<K, T, C, A>* m)
{
    CELER_EXPECT(m);

    for (auto&& kv : *m)
    {
        (*this)(&kv.second);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct a filtering object.
 */
auto ImportDataTrimmer::make_filterer(size_type vec_size) const -> GridFilterer
{
    auto stride = vec_size >= 2 ? vec_size / (options_.max_size - 1) : 1;

    return {stride, vec_size};
}

//---------------------------------------------------------------------------//
/*!
 * Whether to keep the data at this index.
 */
bool ImportDataTrimmer::GridFilterer::operator()(size_type i) const
{
    return i % stride == 0 || i + 1 == orig_size;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
