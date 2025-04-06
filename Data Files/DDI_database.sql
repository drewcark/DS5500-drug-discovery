
create user if not exists 'admin'@'localhost' identified by 'group7';
drop database if exists DDI;
create database DDI;
grant all privileges on *.* to 'admin'@'localhost';
use DDI;
inputs
create table interactions (
	int_code int not null,
    int_desc mediumtext not null,
    primary key (int_code)
);

create table y_codes (
    int_reduced_code int not null,
    int_code int not null,
    primary key (int_reduced_code),
    foreign key (int_code) references interactions(int_code)
);

create table inputs (
    inp_num int not null,
    molecule1 varchar(2800),
    molecule2 varchar(2800),
    result int,
    result_reduced bool,
    primary key (inp_num),
    foreign key (result) references y_codes(int_reduced_code)
);
