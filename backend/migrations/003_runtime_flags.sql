CREATE TABLE runtime_flags (
    flag_name TEXT PRIMARY KEY CHECK (flag_name IN (
        'maintenance', 'read_only', 'signups_enabled', 'max_users'
    )),
    flag_value INTEGER NOT NULL CHECK (
        (flag_name = 'max_users' AND flag_value >= 1)
        OR (flag_name != 'max_users' AND flag_value IN (0, 1))
    )
);
