# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdint cimport uint64_t
from queue import Full

cdef extern from "time.h":
    ctypedef struct timespec "struct timespec":
        long tv_sec
        long tv_nsec
    int clock_gettime(int clock_id, timespec *tp) nogil
    int clock_nanosleep(int clock_id, int flags, const timespec *request, timespec *remaining) nogil

cdef enum:
    _CLOCK_MONOTONIC = 1
    _TIMER_ABSTIME = 1
    _EINTR = 4

cdef int _sleep_until_ns(long long target_ns) noexcept nogil:
    cdef timespec deadline
    cdef int rc
    deadline.tv_sec = <long>(target_ns // 1000000000)
    deadline.tv_nsec = <long>(target_ns % 1000000000)
    while True:
        rc = clock_nanosleep(_CLOCK_MONOTONIC, _TIMER_ABSTIME, &deadline, NULL)
        if rc == 0:
            return 0
        if rc != _EINTR:
            return rc

cdef long long _monotonic_ns() noexcept nogil:
    cdef timespec now
    if clock_gettime(_CLOCK_MONOTONIC, &now) != 0:
        return -1
    return <long long>now.tv_sec * 1000000000 + now.tv_nsec

def produce_arrivals(object work_queue, object arrivals, object drops,
                     int offered_count, int frame_count,
                     long long trial_start_ns, long long period_ns,
                     int omit_sequence=-1):
    cdef int sequence
    cdef int rc
    cdef int depth
    cdef int max_depth = 0
    cdef long long scheduled_ns
    cdef long long arrived_ns
    cdef dict job

    for sequence in range(offered_count):
        scheduled_ns = trial_start_ns + <long long>sequence * period_ns
        with nogil:
            rc = _sleep_until_ns(scheduled_ns)
            arrived_ns = _monotonic_ns()
        if rc != 0:
            raise OSError(rc, "clock_nanosleep failed")
        if arrived_ns < 0:
            raise OSError("clock_gettime(CLOCK_MONOTONIC) failed")
        job = {
            "sequence": sequence,
            "frame_index": sequence % frame_count,
            "scheduled_ns": scheduled_ns,
            "arrived_ns": arrived_ns,
            "queue_depth_before_enqueue": work_queue.qsize(),
        }
        arrivals.append(job)
        if sequence == omit_sequence:
            drops.append({**job, "reason": "injected_missing_frame"})
            continue
        try:
            work_queue.put_nowait(job)
            depth = work_queue.qsize()
            if depth > max_depth:
                max_depth = depth
        except Full:
            drops.append({**job, "queue_depth_at_drop": work_queue.qsize(),
                          "reason": "drop_newest_queue_full"})
    return max_depth


def pack_iq_batch(const uint64_t[:, ::1] source,
                  uint64_t[:, ::1] destination,
                  const Py_ssize_t[::1] indices):
    """Gather raw complex64 words into the frame-last pinned layout."""
    cdef Py_ssize_t row, frame
    cdef Py_ssize_t count = indices.shape[0]
    cdef Py_ssize_t capacity = destination.shape[1]
    if destination.shape[0] != source.shape[0] or count > capacity:
        raise ValueError("incompatible batch dimensions")
    for frame in range(count):
        if indices[frame] < 0 or indices[frame] >= source.shape[1]:
            raise ValueError("frame index outside the source")

    with nogil:
        for row in range(source.shape[0]):
            for frame in range(count):
                destination[row, frame] = source[row, indices[frame]]
            for frame in range(count, capacity):
                destination[row, frame] = 0
