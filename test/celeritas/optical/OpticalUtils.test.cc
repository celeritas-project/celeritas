//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalUtils.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/detail/OpticalUtils.hh"

#include <memory>
#include <numeric>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/action/detail/TrackInitAlgorithms.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class T, MemSpace M>
using StateVal = StateCollection<T, Ownership::value, M>;
template<class T, MemSpace M>
using StateRef = StateCollection<T, Ownership::reference, M>;

template<MemSpace M>
std::vector<int> locate_vacancies(std::vector<TrackStatus> const& input)
{
    if constexpr (M == MemSpace::device)
    {
        device().create_streams(1);
    }

    StateVal<TrackStatus, MemSpace::host> host_status;
    make_builder(&host_status).insert_back(input.begin(), input.end());
    StateVal<TrackStatus, M> status(host_status);

    StateVal<TrackSlotId, M> vacancies;
    resize(&vacancies, status.size());

    StateRef<TrackStatus, M> status_ref(status);
    StateRef<TrackSlotId, M> vacancies_ref(vacancies);
    size_type num_vacancies = optical::detail::copy_if_vacant(
        status_ref, vacancies_ref, StreamId{0});

    auto host_vacancies = copy_to_host(vacancies);

    std::vector<int> result;
    for (auto tid : range(TrackSlotId{num_vacancies}))
    {
        result.push_back(static_cast<int>(host_vacancies[tid].unchecked_get()));
    }
    return result;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(OpticalUtilsTest, find_distribution_index)
{
    using detail::find_distribution_index;

    size_type num_workers = 8;
    std::vector<size_type> work = {1, 1, 5, 2, 5, 8, 1, 6, 7, 7};
    std::vector<size_type> offsets(work.size());
    std::partial_sum(work.begin(), work.end(), offsets.begin());

    static unsigned int const expected_offsets[]
        = {1u, 2u, 7u, 9u, 14u, 22u, 23u, 29u, 36u, 43u};
    EXPECT_VEC_EQ(expected_offsets, offsets);

    LocalWorkCalculator<size_type> calc_local_work{offsets.back(), num_workers};

    std::vector<size_type> result(offsets.back());
    for (auto i : range(num_workers))
    {
        size_type local_work = calc_local_work(i);
        for (auto j : range(local_work))
        {
            size_type result_idx = j * num_workers + i;
            size_type work_idx
                = find_distribution_index(make_span(offsets), result_idx);
            result[result_idx] = work_idx;
        }
    }
    static unsigned int const expected_result[]
        = {0u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 5u,
           5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 7u, 7u, 7u, 7u, 7u, 7u, 8u,
           8u, 8u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 9u, 9u, 9u};
    EXPECT_VEC_EQ(expected_result, result);

    EXPECT_EQ(0, find_distribution_index(make_span(offsets), 0));
    EXPECT_EQ(1, find_distribution_index(make_span(offsets), 1));
    EXPECT_EQ(4, find_distribution_index(make_span(offsets), 13));
    EXPECT_EQ(5, find_distribution_index(make_span(offsets), 14));
    EXPECT_EQ(9, find_distribution_index(make_span(offsets), 42));
}

TEST(OpticalUtilsTest, copy_if_vacant_host)
{
    using TS = TrackStatus;

    std::vector<TrackStatus> status = {
        TS::alive,
        TS::killed,
        TS::alive,
        TS::alive,
        TS::initializing,
        TS::errored,
        TS::alive,
        TS::killed,
    };
    auto vacancies = locate_vacancies<MemSpace::host>(status);

    EXPECT_EQ(4, vacancies.size());
    static int const expected_vacancies[] = {1, 4, 5, 7};
    EXPECT_VEC_EQ(expected_vacancies, vacancies);
}

TEST(OpticalUtilsTest, TEST_IF_CELER_DEVICE(copy_if_vacant_device))
{
    using TS = TrackStatus;

    std::vector<TrackStatus> status = {
        TS::alive,
        TS::alive,
        TS::initializing,
        TS::initializing,
        TS::killed,
        TS::killed,
        TS::alive,
        TS::alive,
    };
    auto vacancies = locate_vacancies<MemSpace::device>(status);

    EXPECT_EQ(4, vacancies.size());
    static int const expected_vacancies[] = {2, 3, 4, 5};
    EXPECT_VEC_EQ(expected_vacancies, vacancies);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
